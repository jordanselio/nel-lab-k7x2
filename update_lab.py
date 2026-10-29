#!/usr/bin/env python3
"""
NEL Player Lab updater.
Downloads the latest nflverse data, rebuilds every stat and projection, and writes NEL_Player_Lab.html.
Run it after the Monday night game (Tuesday is best):   python update_lab.py
Needs Python 3.9+ and:  pip install pandas numpy
The projection model (model.json) was trained on 2021-2025 by train_model.py and is included.
"""
import json, math, re, sys, time, urllib.request, datetime
from pathlib import Path
import numpy as np, pandas as pd

# ------------------------------ settings you may edit ------------------------------
SEASON = 2026
LAST_WEEK = 18                 # last regular-season week
PLAYOFF_WEEKS = (15, 16, 17)   # your fantasy playoff weeks
MATCHUP_STRENGTH_DEFAULT = 1.0 # slider start: 0 = usage only, 1 = adjustments exactly as learned from history
# -----------------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
CACHE = HERE / 'data_cache'; CACHE.mkdir(exist_ok=True)
BASE = 'https://github.com/nflverse/nflverse-data/releases/download'
URLS = {
 'wk':    f'{BASE}/stats_player/stats_player_week_{SEASON}.csv',
 'team':  f'{BASE}/stats_team/stats_team_week_{SEASON}.csv',
 'snaps': f'{BASE}/snap_counts/snap_counts_{SEASON}.csv',
 'inj':   f'{BASE}/injuries/injuries_{SEASON}.csv',
 'pbp':   f'{BASE}/pbp/play_by_play_{SEASON}.csv.gz',
 'games': 'https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv',
}
def fetch(key, required=True):
    url = URLS[key]; dest = CACHE / (key + ('.csv.gz' if url.endswith('.gz') else '.csv'))
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'nel-player-lab'})
            with urllib.request.urlopen(req, timeout=120) as r, open(dest, 'wb') as f: f.write(r.read())
            print(f'  downloaded {key}'); return dest
        except Exception as e:
            print(f'  {key}: attempt {attempt+1} failed ({e})'); time.sleep(2)
    if dest.exists(): print(f'  using cached {key}'); return dest
    if required: sys.exit(f'Could not download {key}. Check your internet connection and try again.')
    return None
def norm(s):
    s = str(s).lower().replace('.', '').replace("'", '').replace('-', ' ')
    s = re.sub(r'\b(jr|sr|ii|iii|iv|v)\b', '', s); return re.sub(r'[^a-z]', '', s)

if not (HERE/'model.json').exists(): sys.exit('model.json is missing. Run:  python train_model.py')
MODEL = json.load(open(HERE/'model.json'))
ANALOG = json.load(open(HERE/'analog.json')) if (HERE/'analog.json').exists() else None
print('Downloading nflverse data...')
paths = {k: fetch(k, required=k in ('wk', 'team', 'games', 'pbp')) for k in URLS}
GAMES = pd.read_csv(paths['games']); GAMES = GAMES[(GAMES.season == SEASON) & (GAMES.game_type == 'REG')].copy()
PB = pd.read_csv(paths['pbp'], low_memory=False); PB = PB[PB.season_type == 'REG']
TEAM_RAW = pd.read_csv(paths['team'])
# A week only counts as finished when the schedule has its scores AND the player stats AND the play-by-play are all in.
# (nflverse posts these files at slightly different times; this keeps a scheduled run from building a half-updated page.)
games_in = GAMES.groupby('week').size(); scored = GAMES.groupby('week').result.apply(lambda s: s.notna().all())
teams_in = TEAM_RAW.groupby('week').team.nunique(); pbp_games = PB.groupby('week').game_id.nunique()
def week_ready(w): return bool(scored.get(w, False)) and teams_in.get(w, 0) >= 2*games_in[w] and pbp_games.get(w, 0) >= games_in[w]
completed = []
for w in sorted(games_in.index):
    if week_ready(w): completed.append(int(w))
    else: break
if not completed: sys.exit('No completed weeks with full data yet this season.')
lagging = [int(w) for w in sorted(games_in.index) if bool(scored.get(w, False)) and int(w) > max(completed)]
if lagging: print(f'  Note: week(s) {lagging} are finished but nflverse has not posted all of their data yet. Using weeks 1 to {max(completed)} for now.')
WK = max(completed) + 1
season_over = WK > LAST_WEEK
print(f'Completed weeks: 1 to {WK-1}.' + ('' if season_over else f' Projecting week {WK} and the rest of the season.'))

W_ALL = pd.read_csv(paths['wk']); W_ALL = W_ALL[W_ALL.position.isin(['QB','RB','WR','TE']) & (W_ALL.week < WK)].copy()
COLS = dict(tgt='targets', rec='receptions', yds='receiving_yards', td='receiving_tds', car='carries', ryd='rushing_yards', rtd='rushing_tds',
            att='attempts', cmp='completions', pyd='passing_yards', ptd='passing_tds', int='passing_interceptions')
for k, v in COLS.items(): W_ALL[k] = W_ALL[v].fillna(0)
W_ALL['ppr'] = W_ALL.fantasy_points_ppr; W_ALL['key'] = W_ALL.player_display_name.map(norm)
TEAM = pd.read_csv(paths['team']); TEAM = TEAM[TEAM.week < WK]
SNAPS = pd.read_csv(paths['snaps']) if paths['snaps'] else None
INJ = pd.read_csv(paths['inj']) if paths['inj'] else None

STATS = {'QB': ['att','cmp','pyd','ptd','int','car','ryd','rtd'], 'RB': ['car','ryd','rtd','tgt','rec','yds','td'],
         'WR': ['tgt','rec','yds','td','car','ryd','rtd'], 'TE': ['tgt','rec','yds','td','car','ryd','rtd']}
DECAY, K_TDBASE = .9, 15
def slots_for(rows, hist):   # depth order uses usage only (targets / touches / attempts), never yards or TDs
    k = hist.groupby('player_id')[['car','tgt','att']].sum(); pos = hist.groupby('player_id').position.first()
    rk = pd.Series(np.where(pos == 'RB', k.car+k.tgt, np.where(pos == 'QB', k.att, k.tgt)), index=k.index)
    r = rows.copy(); r['rk'] = r.player_id.map(rk).fillna(-1)
    r = r.sort_values(['team','week','position','rk','player_id'], ascending=[True, True, True, False, True])
    r['ord'] = r.groupby(['team','week','position']).cumcount()+1
    r['slot'] = np.where(r.position == 'QB', 'QB', r.position+np.minimum(r.ord, np.where(r.position == 'WR', 3, 2)).astype(str)); return r
hist = W_ALL; nTG = hist.groupby(['team','week']).ngroups
h = slots_for(hist, hist); cols = list(COLS.keys())+['ppr']
da = h.groupby(['opponent_team','slot'])[cols].sum(); ng = hist.groupby('opponent_team').week.nunique()
lg = h.groupby('slot')[cols].sum()/nTG; tl = h.groupby('opponent_team')[cols].sum(); ltl = tl.sum()/nTG; posum = h.groupby('position')[cols].sum()
TDYD = {'recv': {p: posum.loc[p,'td']/max(posum.loc[p,'yds'],1) for p in posum.index}, 'rush': {p: posum.loc[p,'rtd']/max(posum.loc[p,'ryd'],1) for p in posum.index},
        'ptd': posum.loc['QB','ptd']/max(posum.loc['QB','pyd'],1)}
def baseline(prow, pos):
    w = DECAY**(WK-1-prow.week.values); w = w/w.sum(); n = len(prow); b = {s: float((prow[s].values*w).sum()) for s in STATS[pos]}
    if pos in ('WR','TE','RB'):
        b['td'] = (n*b['td']+K_TDBASE*b['yds']*TDYD['recv'][pos])/(n+K_TDBASE); b['rtd'] = (n*b['rtd']+K_TDBASE*b['ryd']*TDYD['rush'][pos])/(n+K_TDBASE)
    else: b['ptd'] = (n*b['ptd']+K_TDBASE*b['pyd']*TDYD['ptd'])/(n+K_TDBASE)
    return b

# ------------------------------ league rosters ------------------------------
RJ = json.load(open(HERE/'rosters.json')); MY = RJ['my_team']; roster = {}
for team, t in RJ['teams'].items():
    for n, slot, tag in t['players']: roster[norm(n)] = (team, slot, tag, n)

# ------------------------------ per-player records ------------------------------
print('Building player data...')
snap = {}
if SNAPS is not None:
    SNAPS['key'] = SNAPS.player.map(norm); snap = {(r.key, r.week, r.team): (int(r.offense_snaps), float(r.offense_pct)) for r in SNAPS.itertuples()}
tm = {(r.team, r.week): (int(r.attempts), int(r.carries), int(r.targets)) for r in TEAM.itertuples()}
played = {}
for r in TEAM.itertuples(): played.setdefault(r.team, set()).add(int(r.week))
injm = {}
if INJ is not None:
    for r in INJ.itertuples():
        if isinstance(r.report_status, str): injm[(r.gsis_id, int(r.week))] = (r.report_status, r.report_primary_injury if isinstance(r.report_primary_injury, str) else '')
def r1(x, d=1): return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), d)
players = []
for pid, g in W_ALL.groupby('player_id'):
    g = g.sort_values('week'); key = g.key.iloc[0]
    if g.fantasy_points_ppr.sum() < 5 and key not in roster: continue
    team = g.team.iloc[-1]; ft, slot, _t, _n = roster.get(key, ('Free Agent', '', '', ''))
    weeks = []
    for r in g.itertuples():
        sn = snap.get((key, r.week, r.team)); ta, ca, tt = tm.get((r.team, r.week), (None, None, None))
        weeks.append(dict(w=int(r.week), opp=r.opponent_team, ppr=r1(r.fantasy_points_ppr, 2), std=r1(r.fantasy_points, 2),
            cmp=int(r.completions), att=int(r.attempts), pyd=int(r.passing_yards), ptd=int(r.passing_tds), int=int(r.passing_interceptions), sck=int(r.sacks_suffered),
            car=int(r.carries), ryd=int(r.rushing_yards), rtd=int(r.rushing_tds), tgt=int(r.targets), rec=int(r.receptions), yds=int(r.receiving_yards), td=int(r.receiving_tds),
            ay=r1(r.receiving_air_yards, 0), yac=r1(r.receiving_yards_after_catch, 0), ts=r1(r.target_share, 3), ays=r1(r.air_yards_share, 3), wopr=r1(r.wopr, 3),
            fl=int((r.rushing_fumbles_lost or 0)+(r.receiving_fumbles_lost or 0)+(r.sack_fumbles_lost or 0)),
            sn=sn[0] if sn else None, snp=sn[1] if sn else None, tpa=ta, tca=ca, ttg=tt))
    latest = None
    for wk_ in (WK, WK-1):
        if (pid, wk_) in injm: latest = dict(week=wk_, status=injm[(pid, wk_)][0], injury=injm[(pid, wk_)][1]); break
    players.append(dict(id=pid, name=g.player_display_name.iloc[0], pos=g.position.iloc[0], team=team,
        head=g.headshot_url.iloc[0] if isinstance(g.headshot_url.iloc[0], str) else '', ft=ft, slot=slot, inj=latest,
        byes=[x for x in range(1, WK) if x not in played.get(team, set())], weeks=weeks))
have = {norm(p['name']) for p in players}
missing = [roster[k][3] for k in roster if k not in have]
if missing: print('  Note: no 2026 stats yet for these roster names (spelling or no games):', ', '.join(missing))

data = dict(players=players, teams=list(RJ['teams'].keys()), my=MY, asof=datetime.date.today().strftime('%b %d, %Y'), maxweek=WK-1, week=WK,
            last=LAST_WEEK, playoff=list(PLAYOFF_WEEKS), wdefault=MATCHUP_STRENGTH_DEFAULT)
byid = {p['id']: p for p in players}
slots = ['QB','RB1','RB2','WR1','WR2','WR3','TE1','TE2']
if not season_over:
    print('Projecting...')
    status = {}; report_out = False
    if INJ is not None:
        wi = INJ[INJ.week == WK]
        status = {r.gsis_id: r.report_status for r in wi.itertuples() if isinstance(r.report_status, str)}
    report_out = len(status) > 0      # False until the NFL posts this week's injury report
    if not report_out: print('  Note: this week\'s injury report is not posted yet; using last week\'s report and historical return rates.')
    last = hist.sort_values('week').groupby('player_id').tail(1).set_index('player_id'); rk = {}
    kk = hist.groupby('player_id')[['car','tgt','att']].sum(); pos_of = hist.groupby('player_id').position.first()
    rk = pd.Series(np.where(pos_of == 'RB', kk.car+kk.tgt, np.where(pos_of == 'QB', kk.att, kk.tgt)), index=kk.index)
    slot_now = {}
    for (team, pos), g in last.groupby(['team','position']):
        order = g.assign(rk=g.index.map(rk)).sort_values('rk', ascending=False).index.tolist(); active = [p for p in order if status.get(p) != 'Out']
        for i, pid in enumerate(active): slot_now[pid] = 'QB' if pos == 'QB' else pos+str(min(i+1, 3 if pos == 'WR' else 2))
        for pid in order:
            if pid not in slot_now: slot_now[pid] = 'QB' if pos == 'QB' else pos+str(min(order.index(pid)+1, 3 if pos == 'WR' else 2))
    # defense by slot (for the Defenses tab and the matchup numbers)
    ranks = {}
    for s in slots:
        ppg = {D: (da.loc[(D, s), 'ppr'] if (D, s) in da.index else 0)/ng[D] for D in ng.index}
        order = sorted(ppg, key=lambda d: -ppg[d]); ranks[s] = {d: i+1 for i, d in enumerate(order)}
    def slotrow(D, s):
        r = da.loc[(D, s)] if (D, s) in da.index else pd.Series({k: 0.0 for k in cols}); return {k: round(float(r[k]), 1) for k in r.index}
    defs = {D: {'games': int(ng[D]), 'slots': {s: dict(slotrow(D, s), rank=ranks[s][D]) for s in slots}} for D in ng.index}
    data['lgslot'] = {s: {k: round(float(lg.loc[s, k]), 3) for k in lg.columns} for s in slots}
    # matchup inputs used by the model (same definitions as training):
    # how opposing players did against each defense compared with their OWN averages in their other games
    gg = h.groupby('player_id').ppr.agg(['sum', 'count']); hh = h.join(gg, on='player_id')
    hh['exp'] = (hh['sum']-hh.ppr)/(hh['count']-1); hh = hh[(hh['count'] >= 3) & (hh.exp >= 3)]; hh['ex'] = hh.ppr-hh.exp
    A_slot = hh.groupby(['opponent_team', 'slot']).agg(ex=('ex', 'sum'), ee=('exp', 'sum'), n=('ex', 'size')); L_slot = hh.groupby('slot').exp.mean()
    A_pos = hh.groupby(['opponent_team', 'position']).agg(ex=('ex', 'sum'), ee=('exp', 'sum')); L_pos = hh.groupby('position').exp.mean()
    adj, adjp, adjn, adjg, tr = {}, {}, {}, {}, {}
    for D in ng.index:
        adj[D], adjp[D], adjn[D], adjg[D] = {}, {}, {}, {}
        for s_ in slots:
            if (D, s_) in A_slot.index:
                r = A_slot.loc[(D, s_)]; adj[D][s_] = round(float(r.ex/(r.ee+4*L_slot[s_])), 4); adjn[D][s_] = [int(r.n), round(float(r.ex), 1), round(float(r.ee), 1)]
                rows_ = hh[(hh.opponent_team == D) & (hh.slot == s_)].sort_values('week')
                adjg[D][s_] = [[x.player_display_name, int(x.week), round(float(x.ppr), 1), round(float(x.exp), 1)] for x in rows_.itertuples()]
            else: adj[D][s_] = 0.0; adjn[D][s_] = [0, 0, 0]; adjg[D][s_] = []
        for p_ in ('QB', 'RB', 'WR', 'TE'):
            adjp[D][p_] = round(float(A_pos.loc[(D, p_), 'ex']/(A_pos.loc[(D, p_), 'ee']+4*L_pos[p_])), 4) if (D, p_) in A_pos.index else 0.0
        n = ng[D]; tr[D] = dict(pass_=round(float((tl.loc[D,'yds']/n)/ltl['yds']), 4), rush=round(float((tl.loc[D,'ryd']/n)/ltl['ryd']), 4))
    # team-level defense from play-by-play
    pb = PB[PB.week < WK]; pa = {}
    for r in GAMES[GAMES.week < WK].itertuples():
        if pd.notna(r.home_score): pa.setdefault(r.home_team, []).append(r.away_score); pa.setdefault(r.away_team, []).append(r.home_score)
    plays = pb[pb.play_type.isin(['pass','run'])]
    if 'qb_kneel' in plays: plays = plays[(plays.qb_kneel != 1) & (plays.qb_spike != 1)]
    def team_def(D):
        d = plays[plays.defteam == D]; ps = d[d.play_type == 'pass']; rs = d[d.play_type == 'run']
        att = ps[ps.sack != 1]; comp = att[att.complete_pass == 1]; gms = max(d.game_id.nunique(), 1); third = d[d.down == 3]
        rzd = pb[(pb.defteam == D) & (pb.yardline_100 <= 20)]; rz_n = rzd.groupby(['game_id','drive']).ngroups
        rz_td = rzd[(rzd.touchdown == 1) & rzd.play_type.isin(['pass','run'])].groupby(['game_id','drive']).ngroups
        f = lambda x, n=1: round(float(x), n)
        return dict(games=gms, pts=f(np.mean(pa.get(D, [0]))), pass_yds=f(comp.yards_gained.sum()/gms), rush_yds=f(rs.yards_gained.sum()/gms),
            pass_td=int(ps.pass_touchdown.sum()), rush_td=int(rs.rush_touchdown.sum()), att=f(len(att)/gms), cmp_pct=f(len(comp)/max(len(att), 1)*100),
            ypa=f(comp.yards_gained.sum()/max(len(att), 1), 2), carries=f(len(rs)/gms), ypc=f(rs.yards_gained.sum()/max(len(rs), 1), 2),
            sacks=int(ps.sack.sum()), qb_hits=int(ps.qb_hit.sum()), ints=int(ps.interception.sum()),
            p10=int((comp.yards_gained >= 10).sum()), p20=int((comp.yards_gained >= 20).sum()), p40=int((comp.yards_gained >= 40).sum()),
            r10=int((rs.yards_gained >= 10).sum()), r20=int((rs.yards_gained >= 20).sum()),
            third_pct=f(third.third_down_converted.sum()/max(len(third), 1)*100), rz_td_pct=f(rz_td/max(rz_n, 1)*100),
            epa_pass=f(ps.epa.mean(), 3), epa_rush=f(rs.epa.mean(), 3), sr_pass=f(ps.success.mean()*100), sr_rush=f(rs.success.mean()*100))
    for D in defs: defs[D]['team'] = team_def(D)
    # schedule: every remaining game with the context the model uses
    fut = GAMES[GAMES.week >= WK].copy(); fut['hi'] = (fut.total_line+fut.spread_line)/2; fut['ai'] = (fut.total_line-fut.spread_line)/2
    wkmean = {w: float(np.nanmean(np.concatenate([g.hi.values, g.ai.values]))) for w, g in fut.groupby('week') if g.hi.notna().any()}
    sched, ros = {}, {}
    for g in fut.itertuples():
        dome = int(g.roof in ('dome', 'closed'))
        ih = (g.hi/wkmean[g.week]) if (g.week in wkmean and pd.notna(g.hi)) else None; ia = (g.ai/wkmean[g.week]) if (g.week in wkmean and pd.notna(g.ai)) else None
        ros.setdefault(g.home_team, []).append([int(g.week), g.away_team, True, dome, None if ih is None else round(float(ih), 4)])
        ros.setdefault(g.away_team, []).append([int(g.week), g.home_team, False, dome, None if ia is None else round(float(ia), 4)])
        if g.week == WK and pd.notna(g.spread_line):
            sched[g.home_team] = dict(opp=g.away_team, home=True, spread=-float(g.spread_line), total=float(g.total_line), day=g.gameday)
            sched[g.away_team] = dict(opp=g.home_team, home=False, spread=float(g.spread_line), total=float(g.total_line), day=g.gameday)
        elif g.week == WK:
            sched[g.home_team] = dict(opp=g.away_team, home=True, spread=0.0, total=0.0, day=g.gameday); sched[g.away_team] = dict(opp=g.home_team, home=False, spread=0.0, total=0.0, day=g.gameday)
    cnt = 0
    for pid, p in byid.items():
        pr = hist[hist.player_id == pid]
        if len(pr) == 0: continue
        team = pr.sort_values('week').team.iloc[-1]; p['team'] = team; pos = p['pos']
        slot = slot_now.get(pid, pos if pos == 'QB' else pos+'1'); stt = status.get(pid)
        lastw = max(played.get(team, {0})); inact = not bool((pr.week == lastw).any())
        prev = injm.get((pid, WK-1), (None, ''))[0] if not report_out else None
        outprev = bool(prev == 'Out' and inact)    # listed Out last week and the new week's report is not posted yet
        entry = dict(slot=slot, status=stt, out=(stt == 'Out'), outprev=outprev, gp=int(len(pr)), inact=bool(inact), base={k: round(v, 3) for k, v in baseline(pr, pos).items()})
        if team in sched: s = sched[team]; entry.update(opp=s['opp'], home=s['home'], spread=s['spread'], total=s['total'], day=s['day'])
        else: entry['bye'] = True
        p['proj'] = entry; cnt += 1
    keep = ('center', 'coef_noline', 'coef_line', 'rq', 'avail', 'cv', 'ros_check', 'seasons', 'n_train')
    data.update(defs=defs, sched=sched, ros=ros, adj=adj, adjp=adjp, adjn=adjn, adjg=adjg, tr=tr, model={k: MODEL[k] for k in keep}, analog=ANALOG)
    print(f'  projected {cnt} players; byes this week: {sorted(set(defs) - set(sched)) or "none"}')
else:
    data.update(defs={}, sched={}, ros={}, adj={}, adjp={}, adjn={}, adjg={}, tr={}, model=None, analog=None, lgslot={})

html = (HERE/'lab_template.html').read_text(encoding='utf-8').replace('__DATA__', json.dumps(data, separators=(',', ':')))
out = HERE/'NEL_Player_Lab.html'; out.write_text(html, encoding='utf-8'); print(f'Done. Open {out}')
