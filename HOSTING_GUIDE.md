# Cloud Hosting Guide (100% Free, 24/7)

To get true 24/7 hosting completely for free, we will use **Render** combined with **UptimeRobot**. 

Since we have added a small background web server (`keep_alive.py`) to your bot, it can now be hosted as a Web Service on Render, and UptimeRobot will ping it every 10 minutes so it never goes to sleep!

### Step 1: Upload Your Code to GitHub
1. Create a free account at [GitHub](https://github.com/).
2. Create a new **Private Repository**.
3. Upload the following files to your new repository:
   - `bot.py`
   - `keep_alive.py`
   - `naxcuure(3).xlsx`
   - `requirements.txt`
   
*(Note: Do **NOT** upload the `.env` file to GitHub. Your secrets are safe on your computer thanks to the `.gitignore` file).*

### Step 2: Deploy to Render
1. Create a free account at [Render](https://render.com/).
2. Click **New +** -> **Web Service**.
3. Connect your GitHub account and select the repository you just made.
4. Fill in the settings:
   - **Name**: `my-telebot` (or anything you want)
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python bot.py`
   - **Instance Type**: Free
5. **CRITICAL STEP - ADD SECRETS:** Scroll down to **Environment Variables** and add two variables:
   - Key: `TELEGRAM_BOT_TOKEN` | Value: *(Paste your token here)*
   - Key: `GEMINI_API_KEY` | Value: *(Paste your API key here)*
6. Click **Create Web Service** and wait a few minutes for it to deploy.
7. Once deployed, you will see a URL near the top (e.g., `https://my-telebot.onrender.com`). **Copy this URL**.

### Step 3: Keep It Awake Forever (UptimeRobot)
Render's free tier goes to sleep if no one visits the URL for 15 minutes. We will use UptimeRobot to visit the URL automatically every 10 minutes so it runs 24/7.

1. Go to [UptimeRobot](https://uptimerobot.com/) and create a free account.
2. Click **Add New Monitor**.
3. Set the following:
   - **Monitor Type**: HTTP(s)
   - **Friendly Name**: `Keep Bot Awake`
   - **URL**: Paste the Render URL you copied earlier.
   - **Monitoring Interval**: 10 minutes
4. Click **Create Monitor**.

🎉 **That's it!** Your bot is now running 24/7 forever, completely for free, with zero maintenance required.
