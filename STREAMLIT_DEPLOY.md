# Deploying to Streamlit Community Cloud

This project is ready to be deployed to Streamlit Community Cloud.

## 1. Push to GitHub
Ensure your project is pushed to a public GitHub repository.

1. Initialize git (if not already done):
   ```bash
   git init
   git add .
   git commit -m "Initial commit for Neural Codec Steering"
   ```
2. Create a new repository on GitHub.
3. Push your code:
   ```bash
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
   git branch -M main
   git push -u origin main
   ```

## 2. Deploy on Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io/).
2. Connect your GitHub account.
3. Click "New app".
4. Select your repository, branch (`main`), and main file (`streamlit_app.py`).
5. Click **Deploy**.

## 3. Configuration
Streamlit Cloud will automatically install dependencies from `requirements.txt`.
- The application uses `gpt2-small` and a `sae_lens` artifact. These will be downloaded on the first run.
- `st.cache_resource` is used to ensure the model sits in memory efficiently after the first load.

## Note on Memory
GPT-2 Small + SAE requires < 2GB RAM, which fits comfortably within Streamlit Cloud's free tier availability.
