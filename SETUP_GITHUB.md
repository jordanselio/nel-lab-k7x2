# Put NEL Player Lab on GitHub so it updates itself

You do this once. After that the page rebuilds every Tuesday and Friday and lives at a web link you can open on any device.

## 1. Make the repository
1. Sign in (or sign up, it's free) at github.com.
2. Click **New repository**. Give it a name that is hard to guess, for example `nel-lab-k7x2`. Choose **Public**. Leave "Add a README" unchecked. Click **Create repository**.

## 2. Upload the files
1. Unzip NEL_Lab_GitHub.zip on your computer.
2. In your new repository, click **uploading an existing file**. Drag in everything from the unzipped folder, including the `.github` folder, then click **Commit changes**.
3. If the `.github` folder did not come across (it is hidden on a Mac; press Cmd+Shift+. in Finder to show it), add it by hand:
   - Click **Add file > Create new file**.
   - In the name box type `.github/workflows/update.yml` (typing the slashes makes the folders).
   - Open `workflow_update.yml` from the unzipped folder, copy everything in it, and paste it into the big box.
   - Click **Commit changes**.

## 3. Turn on the web page
1. Open **Settings > Pages**.
2. Under **Build and deployment**, set **Source** to **GitHub Actions**.

## 4. Run it the first time
1. Open the **Actions** tab, click **Update NEL Player Lab** on the left, then **Run workflow**.
2. Wait about 3 minutes for a green check mark.
3. Your page is at `https://YOUR-USERNAME.github.io/YOUR-REPOSITORY-NAME/`. Bookmark it.

## If you already set this up and are just updating
Upload these files over the old ones (Add file > Upload files): `update_lab.py`, `lab_template.html`, `model.json`, `analog.json`, and `.github/workflows/update.yml`. The page rebuilds by itself.

## Every week
- Nothing to do. The page rebuilds on its own Tuesday morning, Tuesday evening, and Friday evening (US time).
- **When rosters change** (trades, waivers): open `rosters.json` in your repository, click the pencil icon, edit, and click **Commit changes**. The page rebuilds within a few minutes. Or send the new rosters to Claude and paste in the file it gives back.
- To refresh right now, go to Actions > Update NEL Player Lab > Run workflow.

## Good to know
- **The page is public.** Anyone with the link can open it, and it shows your league's team names and rosters. A hard-to-guess repository name keeps it low-key, but it is not private. A private page needs a paid GitHub plan.
- **Scheduled runs pause after 60 days with no activity in the repository.** If that happens GitHub shows a banner in the Actions tab with a button to turn it back on. Any edit, such as a roster change, also keeps it going.
- **Runs can start a few minutes late.** That is normal for GitHub.
- **If a run fails** (for example a data download problem), GitHub emails you and the page from the last good run stays up.
- **Next season:** change `SEASON = 2026` to the new year at the top of `update_lab.py`. Rerun `train_model.py` once if you want to refit the model with the 2026 results.
