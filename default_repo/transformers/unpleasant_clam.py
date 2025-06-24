from queue import Queue

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer


@transformer
def transform(messages: Queue, *args, **kwargs):
    """
    Template code for a transformer block.

    Args:
        messages: List of messages in the stream.

    Returns:
        Transformed messages
    """
    new_losses = []
    while not messages.empty():
        update = messages.get()
        if update["type"] == "loss":
            new_losses.append(update["data"]["loss"])
        elif update["type"] == "error":
            print(f"Error: {update['data']}", "Training error occurred")
    
    return new_losses
