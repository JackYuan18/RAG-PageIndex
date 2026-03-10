from pageindex.utils import ChatGPT_API, Free_API

def keywords_are_similar(kw1, kw2,):
    prompt = f"""Your are an expert in determining whether two keywords have similar meanings.
    You are given two keywords: {kw1} and {kw2}.
    
    Return a boolean variable only.
    True if they are similar in meaning, False otherwise.

    IMPORTANT: You must respond with ONLY a single word: either "True" or "False".
    Do not include any explanation, reasoning, or additional text.
    Do not use markdown formatting or code blocks.
    Just respond with: True or False
    """
    response = ChatGPT_API(model='gpt-5.1', prompt=prompt)
    return response


if __name__ == "__main__":
    kw1 = "predictive variance bounds"
    kw2 = "variance bounds"
    similar = keywords_are_similar(kw1, kw2)
    
    print(similar, type(similar))
    if similar:
        print("similar")