#!/usr/bin/env python3
"""
Trains the projection model from past seasons and writes model.json.
You normally only need to run this once (a trained model.json is included).
Re-run it if you want to refit, e.g. next season:   python train_model.py
Needs: pandas numpy
"""
import json, sys, time, urllib.request, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
HERE = Path(__file__).resolve().parent; HIST = HERE/'data_cache'/'hist'; HIST.mkdir(parents=True, exist_ok=True)
SEASONS = [int(x) for x in sys.argv[1:]] or [2021, 2022, 2023, 2024, 2025]
BASE = 'https://github.com/nflverse/nflverse-data/releases/download'
def get(url, dest):
    if dest.exists(): return dest
    for a in range(3):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'nel-player-lab'})
            with urllib.request.urlopen(req, timeout=120) as r, open(dest, 'wb') as f: f.write(r.read())
            print('  downloaded', dest.name); return dest
        except Exception as e: print('  retry', dest.name, e); time.sleep(2)
    sys.exit('download failed: ' + url)
print('Getting past seasons...')
for S in SEASONS:
    get(f'{BASE}/stats_player/stats_player_week_{S}.csv', HIST/f'hist_stats_player_week_{S}.csv')
    get(f'{BASE}/injuries/injuries_{S}.csv', HIST/f'hist_injuries_{S}.csv')
G = pd.read_csv(get('https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv', HIST/'games.csv'))
G = G[G.game_type == 'REG']

COLS = dict(tgt='targets', rec='receptions', yds='receiving_yards', td='receiving_tds', car='carries', ryd='rushing_yards', rtd='rushing_tds',
            att='attempts', cmp='completions', pyd='passing_yards', ptd='passing_tds', int='passing_interceptions')
STATS = {'QB': ['att','cmp','pyd','ptd','int','car','ryd','rtd'], 'RB': ['car','ryd','rtd','tgt','rec','yds','td'],
         'WR': ['tgt','rec','yds','td','car','ryd','rtd'], 'TE': ['tgt','rec','yds','td','car','ryd','rtd']}
def pts(s): return s.get('rec',0)+.1*s.get('yds',0)+6*s.get('td',0)+.1*s.get('ryd',0)+6*s.get('rtd',0)+.04*s.get('pyd',0)+4*s.get('ptd',0)-2*s.get('int',0)
def load_season(S):
    d = pd.read_csv(HIST/f'hist_stats_player_week_{S}.csv', low_memory=False)
    d = d[d.position.isin(['QB','RB','WR','TE']) & (d.season_type == 'REG')].copy()
    for k, v in COLS.items(): d[k] = d[v].fillna(0)
    d['ppr'] = d.fantasy_points_ppr; d['season'] = S; return d
rows = []
for r in G.itertuples():
    for home in (True, False):
        t = r.home_team if home else r.away_team; sp = r.spread_line; tot = r.total_line
        rows.append(dict(season=r.season, week=r.week, team=t, home=int(home), rest=(r.home_rest if home else r.away_rest),
                         dome=int(r.roof in ('dome','closed')), imp=((tot+sp)/2 if home else (tot-sp)/2)))
CTX = pd.DataFrame(rows).set_index(['season','week','team'])
IMPMEAN = pd.DataFrame(rows).groupby(['season','week']).imp.mean()

def slots_for(rows, hist):
    k = hist.groupby('player_id')[['car','tgt','att']].sum(); pos = hist.groupby('player_id').position.first()
    rk = pd.Series(np.where(pos == 'RB', k.car+k.tgt, np.where(pos == 'QB', k.att, k.tgt)), index=k.index)
    r = rows.copy(); r['rk'] = r.player_id.map(rk).fillna(-1)
    r = r.sort_values(['team','week','position','rk','player_id'], ascending=[True, True, True, False, True])
    r['ord'] = r.groupby(['team','week','position']).cumcount()+1
    r['slot'] = np.where(r.position == 'QB', 'QB', r.position+np.minimum(r.ord, np.where(r.position == 'WR', 3, 2)).astype(str)); return r
def features(df, W, min_prior=3):
    hist = df[df.week < W]; cur = df[df.week == W]
    if len(hist) == 0 or len(cur) == 0: return None
    h = slots_for(hist, hist); cur = slots_for(cur, hist); nTG = hist.groupby(['team','week']).ngroups; ng = hist.groupby('opponent_team').week.nunique()
    cols = list(COLS.keys())+['ppr']; da = h.groupby(['opponent_team','slot'])[cols].sum(); lg = h.groupby('slot')[cols].sum()/nTG
    tl = h.groupby('opponent_team')[cols].sum(); ltl = tl.sum()/nTG; pos = h.groupby('position')[cols].sum()
    tdr = {p: pos.loc[p,'td']/max(pos.loc[p,'yds'],1) for p in pos.index}; tdu = {p: pos.loc[p,'rtd']/max(pos.loc[p,'ryd'],1) for p in pos.index}; tdp = pos.loc['QB','ptd']/max(pos.loc['QB','pyd'],1)
    gg = h.groupby('player_id').ppr.agg(['sum', 'count']); hh = h.join(gg, on='player_id')
    hh['exp'] = (hh['sum']-hh.ppr)/(hh['count']-1)            # his average in his OTHER games before this week
    hh = hh[(hh['count'] >= 3) & (hh.exp >= 3)]; hh['ex'] = hh.ppr-hh.exp
    A_slot = hh.groupby(['opponent_team', 'slot']).agg(ex=('ex', 'sum'), ee=('exp', 'sum')); L_slot = hh.groupby('slot').exp.mean()
    A_pos = hh.groupby(['opponent_team', 'position']).agg(ex=('ex', 'sum'), ee=('exp', 'sum')); L_pos = hh.groupby('position').exp.mean()
    hg = {pid: g.sort_values('week') for pid, g in hist.groupby('player_id')}; out = []
    for r in cur.itertuples():
        g = hg.get(r.player_id)
        if g is None or len(g) < min_prior: continue
        n = len(g); w = .9**(W-1-g.week.values); w = w/w.sum(); P = r.position; base = {s: float((g[s].values*w).sum()) for s in STATS[P]}
        if P == 'QB': base['ptd'] = (n*base['ptd']+15*base['pyd']*tdp)/(n+15)
        else:
            base['td'] = (n*base['td']+15*base['yds']*tdr[P])/(n+15); base['rtd'] = (n*base['rtd']+15*base['ryd']*tdu[P])/(n+15)
        D = r.opponent_team; nD = ng.get(D, 0)
        if nD == 0 or r.slot not in lg.index: continue
        L = lg.loc[r.slot, 'ppr']
        try: A = da.loc[(D, r.slot), 'ppr']
        except KeyError: A = 0.0
        c = CTX.loc[(r.season, W, r.team)]
        out.append(dict(season=r.season, week=W, pos=P, actual=r.ppr, base=pts(base), savg=g.ppr.mean(), nD=nD,
            **{('b_'+k): v for k, v in base.items()}, **{('a_'+k): float(getattr(r, k)) for k in STATS[P]},
            adj_slot4=((A_slot.loc[(D, r.slot), 'ex']/(A_slot.loc[(D, r.slot), 'ee']+4*L_slot[r.slot])) if (D, r.slot) in A_slot.index else 0.0),
            adj_pos4=((A_pos.loc[(D, P), 'ex']/(A_pos.loc[(D, P), 'ee']+4*L_pos[P])) if (D, P) in A_pos.index else 0.0), pass_ratio=((tl.loc[D,'yds']/nD)/ltl['yds']), rush_ratio=((tl.loc[D,'ryd']/nD)/ltl['ryd']),
            home_c=c.home-.5, dome=c.dome, imp_ratio=c.imp/IMPMEAN.loc[(r.season, W)]))
    return pd.DataFrame(out)
print('Building training rows (a few minutes)...')
parts = []
for S in SEASONS:
    df = load_season(S)
    for W in range(4, int(df.week.max())+1):
        f = features(df, W)
        if f is not None: parts.append(f)
    print(' ', S, 'done', flush=True)
D = pd.concat(parts); D = D[D.savg >= 4]
print('training rows:', len(D))

# matchup measure = how opposing players did against this defense compared with their OWN averages
FEATS = {'QB': ['adj_slot4','home_c','dome'], 'RB': ['adj_slot4','rush_ratio','home_c','dome'], 'WR': ['adj_pos4','home_c','dome'], 'TE': ['adj_pos4','pass_ratio','home_c','dome']}
CENTER = {'adj_slot4': 0, 'adj_pos4': 0, 'pass_ratio': 1, 'rush_ratio': 1, 'home_c': 0, 'dome': 0, 'imp_ratio': 1}
def design(df, feats): return np.column_stack([df.base.values*(df[f].values-CENTER[f]) for f in feats])
def fit(df, feats, lam=1.0):
    X = design(df, feats); y = (df.actual-df.base).values; return np.linalg.solve(X.T@X+lam*np.eye(X.shape[1]), X.T@y)
model = dict(center=CENTER, coef_noline={}, coef_line={}, rq={}, feats_noline=FEATS, feats_line={p: f+['imp_ratio'] for p, f in FEATS.items()})
mae = lambda a, b: float(np.abs(a-b).mean())
cv = []
for p, f in FEATS.items():
    s = D[D.pos == p]; model['coef_noline'][p] = dict(zip(f, [round(float(x), 4) for x in fit(s, f)]))
    fl = model['feats_line'][p]; model['coef_line'][p] = dict(zip(fl, [round(float(x), 4) for x in fit(s, fl)]))
    mu = s.base.values+design(s, f)@np.array(list(model['coef_noline'][p].values())); r = (s.actual/mu)[mu > 3]
    model['rq'][p] = {k: round(float(np.quantile(r, q)), 3) for k, q in (('p10', .1), ('p25', .25), ('p75', .75), ('p90', .9))}
# leave-one-season-out check
for S in SEASONS:
    tr = D[D.season != S]; te = D[D.season == S]; b_all = []; m_all = []; a_all = []
    for p, f in FEATS.items():
        t = te[te.pos == p]; b = fit(tr[tr.pos == p], f); m_all.append(t.base.values+design(t, f)@b); b_all.append(t.base.values); a_all.append(t.actual.values)
    a, b, m = map(np.concatenate, (a_all, b_all, m_all)); cv.append((S, mae(a, b), mae(a, m)))
    print(f'  held-out {S}: usage-only error {mae(a, b):.3f} -> with matchup model {mae(a, m):.3f}')
model['cv'] = [dict(season=int(s), usage=round(u, 3), model=round(m, 3)) for s, u, m in cv]
model['n_train'] = int(len(D)); model['seasons'] = SEASONS

# ---------------- availability tables ----------------
print('Computing availability...')
rows = []; stat_rows = []
for S in SEASONS:
    df = load_season(S); inj = pd.read_csv(HIST/f'hist_injuries_{S}.csv'); inj = inj[inj.position.isin(['QB','RB','WR','TE'])]
    gp = G[G.season == S]; sched = set()
    for r in gp.itertuples(): sched.add((r.home_team, r.week)); sched.add((r.away_team, r.week))
    appear = set(zip(df.player_id, df.week)); avgs = {}
    for pid, g in df.groupby('player_id'):
        g = g.sort_values('week'); pos = g.position.iloc[0]; t = g.team.iloc[-1]; wk = set(g.week)
        for cut in range(5, 18):
            prior = g[g.week < cut]
            if len(prior) < 4 or prior.ppr.mean() < 6: continue
            last = (cut-1) in wk
            for h in range(1, 18):
                W = cut+h-1
                if W > 18 or (t, W) not in sched: continue
                rows.append((pos, h, int(last), int(W in wk)))
    inj['played'] = [int((a, b) in appear) for a, b in zip(inj.gsis_id, inj.week)]
    rel = {pid for pid, g in df.groupby('player_id') if g.ppr.mean() >= 6}
    inj = inj[inj.gsis_id.isin(rel)]
    st = inj[['gsis_id','week','report_status','played']].dropna(subset=['report_status']); st['season'] = S; stat_rows.append(st)
    # recovery after being listed Out at week c: appear at c+k?
    outs = st[st.report_status == 'Out']
    for r in outs.itertuples():
        for k in range(1, 9):
            tt = df[df.player_id == r.gsis_id].team
            if len(tt) and (tt.iloc[0], r.week+k) in sched: stat_rows.append(pd.DataFrame([dict(kind='out_rec', k=k, played=int((r.gsis_id, r.week+k) in appear))]))
a = pd.DataFrame(rows, columns=['pos','h','last','played'])
avail = {'healthy': {p: [round(float(a[(a.pos == p) & (a.h == h) & (a['last'] == 1)].played.mean()), 3) for h in range(1, 18)] for p in ('QB','RB','WR','TE')},
         'inactive': {p: [round(float(a[(a.pos == p) & (a.h == h) & (a['last'] == 0)].played.mean()), 3) for h in range(1, 18)] for p in ('QB','RB','WR','TE')}}
S_ = pd.concat([x for x in stat_rows if 'report_status' in x.columns])
avail['status'] = {k: round(float(S_[S_.report_status == k].played.mean()), 3) for k in ('Questionable', 'Doubtful', 'Out')}
O_ = pd.concat([x for x in stat_rows if 'kind' in x.columns])
avail['out_recovery'] = [round(float(O_[O_.k == k].played.mean()), 3) for k in range(1, 9)]
# fill any gaps (nan) with last good value
for grp in ('healthy', 'inactive'):
    for p, v in avail[grp].items():
        for i in range(len(v)):
            if v[i] != v[i]: v[i] = v[i-1] if i else 0.9
model['avail'] = avail

# ---------------- rest-of-season check: does counting availability help? ----------------
print('Checking rest-of-season totals...')
rat = {p: (D[D.pos == p].actual/D[D.pos == p].base)[D[D.pos == p].base > 2].values for p in ('QB', 'RB', 'WR', 'TE')}
rng = np.random.default_rng(1); chk2 = []
chk = []
for S in SEASONS:
    df = load_season(S); sched = {}
    for r in G[G.season == S].itertuples(): sched.setdefault(r.home_team, set()).add(r.week); sched.setdefault(r.away_team, set()).add(r.week)
    for cut in (8, 10, 12):
        for pid, g in df.groupby('player_id'):
            g = g.sort_values('week'); pos = g.position.iloc[0]; t = g.team.iloc[-1]; pr = g[g.week < cut]
            if len(pr) < 4 or pr.ppr.mean() < 6 or (cut-1) not in set(pr.week): continue
            w = .9**(cut-1-pr.week.values); w = w/w.sum(); b = (pr.ppr.values*w).sum()
            wks = [W for W in range(cut, 19) if W in sched.get(t, set())]
            if len(wks) < 4: continue
            Ps = np.array([avail['healthy'][pos][min(W-cut, 16)] for W in wks]); act = float(g[g.week.isin(wks)].ppr.sum())
            chk.append((act, b*len(wks), float((Ps*b).sum())))
            sim = ((rng.random((300, len(wks))) < Ps)*(rng.choice(rat[pos], (300, len(wks)))*b)).sum(1); chk2.append((pos, act, float((Ps*b).sum()), float(sim.std())))
c = np.array(chk)
model['ros_check'] = dict(n=int(len(c)), mae_ignore=round(float(np.abs(c[:, 0]-c[:, 1]).mean()), 1), mae_with=round(float(np.abs(c[:, 0]-c[:, 2]).mean()), 1),
                          bias_ignore=round(float((c[:, 1]-c[:, 0]).mean()), 1), bias_with=round(float((c[:, 2]-c[:, 0]).mean()), 1))
print('  ', model['ros_check'])
# how much do rest-of-season totals vary beyond injuries and weekly luck? (role changes, slumps, breakouts, long injuries)
C2 = pd.DataFrame(chk2, columns=['pos', 'act', 'pred', 'simsd'])
model['weekly_sd'] = {p: round(float(np.std(rat[p])), 3) for p in rat}
model['persist_sd'] = {}
for p, g_ in C2.groupby('pos'):
    var_act = (((g_.act-g_.pred)/g_.pred)**2).mean(); var_sim = ((g_.simsd/g_.pred)**2).mean()
    model['persist_sd'][p] = round(float(max(var_act-var_sim, 0)**.5), 3)
print('  weekly luck sd:', model['weekly_sd'], ' extra season-long sd:', model['persist_sd'])
json.dump(model, open(HERE/'model.json', 'w'), indent=1)

# ---------------- "similar games" odds data ----------------
print('Building the odds data...')
AF = {'RB': ['ryd','car','tdt','yds','rec'], 'WR': ['yds','rec','tgt','td','ryd'], 'TE': ['yds','rec','tgt','td'], 'QB': ['pyd','ptd','ryd','rtd','int']}
AO = {'RB': ['ryd','rtd','car','tgt','rec','yds','td','tdt','ytot','ppr'], 'WR': ['tgt','rec','yds','td','car','ryd','rtd','tdt','ytot','ppr'],
      'TE': ['tgt','rec','yds','td','car','ryd','rtd','tdt','ytot','ppr'], 'QB': ['att','cmp','pyd','ptd','int','car','ryd','rtd','tdt','ppr']}
def derive(df, pos, pre):
    g = lambda k: df[pre+k] if pre+k in df else 0.0
    if pos == 'QB': return g('ptd')+g('rtd')
    return g('rtd')+g('td')
analog = {}; tabs = {}
for p in ('RB', 'WR', 'TE', 'QB'):
    s_ = D[D.pos == p].copy(); X = pd.DataFrame({f: (derive(s_, p, 'b_') if f == 'tdt' else s_['b_'+f]) for f in AF[p]})
    Y = pd.DataFrame({o: (derive(s_, p, 'a_') if o == 'tdt' else (s_['a_ryd']+s_['a_yds'] if o == 'ytot' else (s_['actual'] if o == 'ppr' else s_['a_'+o]))) for o in AO[p]})
    tabs[p] = (X.reset_index(drop=True), Y.reset_index(drop=True), s_.season.values)
    mean = X.mean(); sd = X.std()+1e-9
    rnd = lambda v, d: (int(round(v)) if d == 0 else round(float(v), d))
    rows = [[round(float(v), 2) for v in xr] + [rnd(v, 1 if o == 'ppr' else 0) for o, v in zip(AO[p], yr)] for xr, yr in zip(X.values, Y.values)]
    analog[p] = dict(feat=AF[p], out=AO[p], mean=[round(float(v), 4) for v in mean], sd=[round(float(v), 4) for v in sd], rows=rows)
# how well do the odds match reality? (each season is predicted using only the other seasons)
def knn_p(Xt, yt, Xq, k=300):
    mu = Xt.mean(); sd = Xt.std()+1e-9; A = ((Xt-mu)/sd).values; Q = ((Xq-mu)/sd).values; out = np.empty(len(Q))
    for i, q in enumerate(Q):
        d2 = ((A-q)**2).sum(1); idx = np.argpartition(d2, k)[:k]; out[i] = yt[idx].mean()
    return out
Xr, Yr, ss = tabs['RB']; ev = ((Yr.ryd >= 50) & (Yr.tdt >= 1)).values.astype(float); pr = np.zeros(len(ev))
for S in SEASONS:
    te = ss == S; pr[te] = knn_p(Xr[~te], ev[~te], Xr[te])
q = pd.qcut(pr, 5, duplicates='drop'); cal = pd.DataFrame({'p': pr, 'y': ev}).groupby(q).agg(p=('p', 'mean'), y=('y', 'mean'))
analog['check'] = dict(event='RB with 50+ rushing yards and a touchdown', base_rate=round(float(ev.mean()), 3), pairs=[[round(float(a), 3), round(float(b), 3)] for a, b in zip(cal.p, cal.y)])
json.dump(analog, open(HERE/'analog.json', 'w'), separators=(',', ':'))
print('Saved analog.json', round((HERE/'analog.json').stat().st_size/1024), 'KB;', 'calibration pairs', analog['check']['pairs'])

print('Saved model.json'); print(json.dumps({k: model[k] for k in ('coef_noline', 'coef_line')}, indent=0)[:900]); print('availability healthy h=1,5,10:', {p: [v[0], v[4], v[9]] for p, v in avail['healthy'].items()})
print('status:', avail['status'], ' out recovery k=1..8:', avail['out_recovery'])
