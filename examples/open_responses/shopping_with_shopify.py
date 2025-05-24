import os
import re
from openai import OpenAI

"""This example demonstrates how to use the OpenAI API to interact with a Shopify store
using the Multi-Cloud Provider (MCP) integration with streaming enabled."""


def main():
    # Initialize the OpenAI client with your API key
    client = OpenAI(
        # Retrieve API key from environment variable
        api_key=os.environ.get("OPENAI_API_KEY"),
    )

    model = "groq@meta-llama/llama-4-maverick-17b-128e-instruct"
    stream = client.responses.create(
        model=model,
        tools=[
            {
                "type": "mcp",
                "server_label": "shopify",
                "server_url": "https://pitchskin.com/api/mcp",
                "allowed_tools": ["search_shop_catalog", "update_cart"]
            }
        ],
        input="Add the Blemish Toner Pads to my cart",
        stream=True
    )

    # Event patterns to match
    event_patterns = [
        r"^response\.created$",
        r"^response\.([^_]+)_([^.]+)\.in_progress$",
        r"^response\.([^_]+)_([^.]+)\.completed$",
        r"^response\.completed$"
    ]

    # Counter for stages
    stage_counter = 1

    # Process the stream events
    for event in stream:
        try:
            # Get the event type safely
            event_type = str(event.type) if hasattr(event, 'type') else ""

            # Match against each pattern
            for pattern in event_patterns:
                match = re.match(pattern, event_type)
                if match:
                    if event_type == "response.created":
                        print(f"{stage_counter}. Started with model {model}")
                        stage_counter += 1
                    elif "in_progress" in event_type:
                        # Extract service and operation from the event type
                        service, operation = match.groups()
                        print(f"{stage_counter}. Executing {service.upper()} {operation}")
                        stage_counter += 1
                    elif "completed" in event_type and event_type != "response.completed":
                        # Extract service and operation from the event type
                        service, operation = match.groups()
                        print(f"{stage_counter}. Finished {service.upper()} {operation}")
                        stage_counter += 1
                    elif event_type == "response.completed" and hasattr(event, 'response'):
                        # Get the actual model used from the response if available
                        response_model = model
                        if hasattr(event, 'response') and hasattr(event.response, 'model'):
                            response_model = event.response.model

                        print(f"{stage_counter}. Completed with model {response_model}")
                        stage_counter += 1

                        # Extract and print the assistant's response
                        try:
                            response = event.response
                            if hasattr(response, 'output') and response.output:
                                output_message = response.output[0]
                                if hasattr(output_message, 'content') and output_message.content:
                                    content_item = output_message.content[0]
                                    if hasattr(content_item, 'text'):
                                        print(f"{stage_counter}. Assistant => {content_item.text}")
                        except Exception as e:
                            print(f"Error extracting final text: {e}")

                    break
        except Exception as e:
            print(f"Error processing event: {e}")
            continue


if __name__ == "__main__":
    main()
