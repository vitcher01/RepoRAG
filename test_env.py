from dotenv import load_dotenv
import os

load_dotenv()

gitlab_url = os.getenv("GITLAB_URL")
gitlab_token = os.getenv("GITLAB_TOKEN")
openai_key = os.getenv("OPENAI_API_KEY")

print(f"✓ GitLab URL: {gitlab_url}")
print(f"✓ GitLab Token: {gitlab_token[:10]}..." if gitlab_token else "✗ No GitLab token")
print(f"✓ OpenAI Key: {openai_key[:10]}..." if openai_key else "✗ No OpenAI key")