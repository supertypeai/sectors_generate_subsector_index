from langchain.chat_models import init_chat_model

from sgx_description.utils.config import GROQ_API_KEY


def get_model(model_name: str = 'gpt-oss-120b'):
    try:
        initiate_model = init_chat_model(
            model_name,  # 'gpt-oss-20b'
            model_provider='groq',
            temperature=0.6,
            max_retries=3,
            api_key=GROQ_API_KEY,
            max_tokens=18000
        ) 

        return initiate_model

    except Exception as error:
        print(f'Error initialize llm: {error}')
        return None  
      
