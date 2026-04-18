# GitHub Repository Setup Guide
## Malaria Cell Detection Project

---

## Step 1 — Create the Repository on GitHub

1. Go to https://github.com/new
2. Set these fields:
   - **Repository name:** `malaria-detection`
   - **Description:** `Hybrid Deep Learning (CNN-BiLSTM) for automated malaria cell detection from blood smear images | B.Tech Final Year Project, MITS 2025`
   - **Visibility:** Public ✅
   - **Initialize with README:** ❌ NO (we have our own)
   - **Add .gitignore:** ❌ NO (we have our own)
3. Click **Create repository**

---

## Step 2 — Extract and Set Up Locally

1. Download and extract `malaria-detection.zip`
2. Open terminal in the extracted `malaria-detection/` folder

```bash
# Verify you're in the right folder
ls
# Should show: README.md  src/  app/  notebooks/  requirements.txt  etc.
```

---

## Step 3 — Initialize Git and Push

```bash
# Initialize git
git init

# Add all files
git add .

# First commit
git commit -m "Initial commit: Malaria detection project - CNN, VGG19, ResNet50, Hybrid CNN-BiLSTM"

# Connect to your GitHub repo (replace with your actual URL)
git remote add origin https://github.com/nitish-reddy25/malaria-detection.git

# Push
git branch -M main
git push -u origin main
```

---

## Step 4 — Add Topics/Tags to the Repo

On GitHub, click the ⚙️ gear icon next to "About" on your repo page.

Add these topics:
```
deep-learning  malaria-detection  computer-vision  cnn  tensorflow
medical-imaging  transfer-learning  lstm  flask  python
```

---

## Step 5 — Pin the Repo on Your Profile

1. Go to https://github.com/nitish-reddy25
2. Click **Customize your pins**
3. Select `malaria-detection` → Save

---

## Step 6 — Update the README with Real Results

After you train the models and get actual numbers, update the Results table in `README.md`:

```markdown
| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Custom CNN | XX.XX% | 0.XXX | 0.XXX | 0.XXX |
...
```

Also replace placeholder links in the Featured Projects section of your **profile README** with:
```markdown
[Malaria Detection](https://github.com/nitish-reddy25/malaria-detection)
```

---

## Step 7 — Add Screenshots to README (Optional but Impactful)

Once you have results:

1. Create a folder: `assets/`
2. Add screenshots: confusion matrices, training curves, web app UI
3. Reference in README:

```markdown
![Confusion Matrix](assets/confusion_matrix_hybrid.png)
![Web App](assets/webapp_screenshot.png)
```

---

## Common Git Commands You'll Use

```bash
# After making changes
git add .
git commit -m "Add training results and confusion matrices"
git push

# Check status
git status

# View history
git log --oneline
```

---

## Repository Checklist

- [ ] Repo created at github.com/nitish-reddy25/malaria-detection
- [ ] All files pushed (src/, app/, notebooks/, README.md)
- [ ] Topics/tags added
- [ ] Repo pinned on profile
- [ ] Profile README updated with real link to this repo
- [ ] Dataset downloaded from Kaggle (not committed to Git)
- [ ] Models trained locally and .h5 files saved to results/saved_models/
