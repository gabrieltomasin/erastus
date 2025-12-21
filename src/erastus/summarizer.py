import requests

class DeepSeekSummarizer:
    def __init__(self, api_key, api_url):
        self.api_key = api_key
        self.api_url = api_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def create_summary_prompt(self, transcript_text, additional_context: str = ""):
        """Create a summarization prompt that first detects transcript language and
        asks the LLM to produce the summary in that same language.

        The output should be structured and include the following sections:
        1) General session summary (1-2 paragraphs)
        2) Main events (bulleted list)
        3) Important player decisions
        4) Discoveries & revelations
        5) Hooks / seeds for the next session
        """

        base_prompt = f"""
You are an expert TTRPG Game Master and a helpful summarizer assistant.

First, detect the primary language used in the session transcript below.

Then, generate a detailed session summary in the same language you detected. Structure the summary exactly with the sections below and keep the content concise and useful for both players and the game master.

Required structure (produce these headings in the detected language):
1. GENERAL SESSION SUMMARY (1-2 paragraphs)
2. MAIN EVENTS (bullet points)
3. IMPORTANT PLAYER DECISIONS
4. DISCOVERIES AND REVELATIONS
5. HOOKS FOR THE NEXT SESSION

Here is the session transcript to analyze:

{transcript_text}

Additional context (if any):
{additional_context}

Write the complete structured summary below, in the detected language.
"""
        
        return base_prompt
    
    def summarize(self, transcript_text, max_tokens=2000):
        """Send the transcript to the configured summarizer API and return the summary text.

        Args:
            transcript_text: Full transcript to summarize.
            max_tokens: Maximum tokens for the LLM response.
        Returns:
            The content returned by the API as a string.
        """
        prompt = self.create_summary_prompt(transcript_text)
        
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": 0.7
        }
        
        try:
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
            else:
                raise Exception(f"API Error: {response.status_code} - {response.text}")
        
        except Exception as e:
            raise Exception(f"Error when calling summarizer API: {str(e)}")