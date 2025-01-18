import requests
from together import Together
import os
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("TOGETHER_API")

client = Together(api_key=api_key)

class ConversationContext:
    """
    A class to maintain context across multiple messages in a conversation.
    Stores the cryptocurrency queries and responses.
    """
    def __init__(self):
        self.context = {}

    def update_context(self, crypto_name, response):
        self.context[crypto_name] = response

    def get_context(self, crypto_name):
        return self.context.get(crypto_name, None)


def get_crypto_price(crypto_name):
    """
    Fetch the current price of the specified cryptocurrency using the CoinCap API.
    """
    try:
        # Construct the API URL dynamically based on the cryptocurrency name
        url = f"https://api.coincap.io/v2/assets/{crypto_name.lower()}"
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        data = response.json()
        price = data['data']['priceUsd']
        return f"The current price of {crypto_name.capitalize()} is ${float(price):,.2f}."
    except requests.exceptions.RequestException as e:
        return f"Error fetching cryptocurrency prices: {e}"
    except KeyError:
        return f"Could not retrieve price for {crypto_name}. Please check the cryptocurrency name."


def llm_agent(crypto_name, context_manager):
    """
    Use Together AI's LLM to generate a response mentioning the cryptocurrency price.
    Maintains context across multiple queries.
    """
    # Check if the price is already in context
    cached_response = context_manager.get_context(crypto_name)
    if cached_response:
        return f"(From context) {cached_response}"

    # Fetch the cryptocurrency price
    crypto_price = get_crypto_price(crypto_name)

    # Generate a response using the LLM
    response = client.completions.create(
        model="meta-llama/Meta-Llama-3-8B-Instruct-Turbo",
        prompt=f"Provide a concise response for {crypto_price}. Do not hallucinate.",
        max_tokens=40,
    )
    llm_response = response.choices[0].text.strip()

    # Update context
    context_manager.update_context(crypto_name, llm_response)
    return llm_response


# Streamlit App
def main():
    st.title("Cryptocurrency Price Checker with LLM Agent")
    st.write("Enter the name of a cryptocurrency to get its current price. System responses will remain in English.")

    context_manager = ConversationContext()
    user_input = st.text_input("Enter cryptocurrency name (or type 'exit' to quit):")

    if user_input.lower() == "exit":
        st.write("Goodbye!")
    elif user_input.lower() == "change language":
        st.write("Language change is not currently supported. System responses will remain in English.")
    elif user_input:
        with st.spinner("Fetching data..."):
            result = llm_agent(user_input, context_manager)
        st.success("Response:")
        st.write(result)

    st.write("### Conversation Context")
    if context_manager.context:
        for crypto, response in context_manager.context.items():
            st.write(f"**{crypto.capitalize()}:** {response}")

if __name__ == "__main__":
    main()
