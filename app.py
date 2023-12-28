import os
import errno
import tempfile
from dotenv import load_dotenv
from flask import Flask
from flask import request
from flask import abort
from linebot import LineBotApi
from linebot import WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent
from linebot.models import TextMessage
from linebot.models import ImageMessage
from linebot.models import TextSendMessage
from linebot.models import FollowEvent
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Initialize Line Bot API
line_bot_api = LineBotApi(os.environ["ACCESS_TOKEN"])
# Initialize Webhook Handler
handler = WebhookHandler(os.environ["CHANNEL_SECRET"])
# Get developer ID from environment variables
developer_id = os.environ["DEVELOPER_ID"]

# Define path for temporary static files
static_tmp_path = os.path.join(os.path.dirname(__file__), "static", "tmp")

# Create directory for temporary static files if it doesn't exist
try:
    os.makedirs(static_tmp_path)
except OSError as exc:
    if exc.errno == errno.EEXIST and os.path.isdir(static_tmp_path):
        pass
    else:
        raise


# Define route for callback
@app.route("/callback", methods=["POST"])
def callback():
    # Get signature from request headers
    signature = request.headers["x-line-signature"]

    # Get request body
    body = request.get_data(as_text=True)
    # Log request body
    app.logger.info("Request body: " + body)

    # Handle the request body and signature
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        # Abort with status 400 if signature is invalid
        abort(400)

    # Return OK if successful
    return "OK"


# Define handler for text messages
@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    # Get text from message event
    text = event.message.text
    # Reply with "Good Morning" if text is "Good Morning"
    if text == "Good Morning":
        line_bot_api.reply_message(
            event.reply_token, TextSendMessage(text="Good Morning")
        )
    # Reply with "Hello" if text is "Hello"
    elif text == "Hello":
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text="Hello"))
    # Reply with "Recieve Message" for all other texts
    else:
        line_bot_api.reply_message(
            event.reply_token, TextSendMessage(text="Recieve Message")
        )


# Load model
model = load_model("my_model.h5")


# Define handler for image messages
@handler.add(MessageEvent, message=ImageMessage)
def handle_content_message(event):
    # Get content of image message
    message_content = line_bot_api.get_message_content(event.message.id)
    # Write content to temporary file
    with tempfile.NamedTemporaryFile(
        dir=static_tmp_path, prefix="jpg" + "-", delete=False
    ) as tf:
        for chunk in message_content.iter_content():
            tf.write(chunk)
            tempfile_path = tf.name

    # Define path for image file
    dist_path = f"{tempfile_path}.jpg"
    # Get name of image file
    dist_name = os.path.basename(dist_path)
    # Rename temporary file to image file
    os.rename(tempfile_path, dist_path)

    # Define path for image file in static directory
    filepath = os.path.join("static", "tmp", dist_name)

    # Load image and convert to array
    img = image.load_img(filepath, target_size=(32, 32))
    img = image.img_to_array(img)
    # Create array of image data
    data = np.array([img])
    # Predict class of image
    result = model.predict(data)
    # Get index of predicted class
    predicted = result.argmax()

    # Define class labels
    class_label = [
        "airplane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "track",
    ]
    # Create answer string
    pred_answer = "This is " + class_label[predicted] + "."

    # Reply with predicted class
    line_bot_api.reply_message(event.reply_token, TextSendMessage(text=pred_answer))


# Define handler for follow events
@handler.add(FollowEvent)
def handle_follow(event):
    # Get profile of user who added as friend
    profile = line_bot_api.get_profile(event.source.user_id)
    # Send message to developer with user's profile information
    line_bot_api.push_message(
        developer_id,
        TextSendMessage(
            text="display:{}\nUserID:{}\nURL:{}\nStatusMessage:{}".format(
                profile.display_name,
                profile.user_id,
                profile.picture_url,
                profile.status_message,
            )
        ),
    )

    # Reply to user with thank you message
    line_bot_api.reply_message(
        event.reply_token, TextSendMessage(text="Thank you for adding as a friend.")
    )


# Run app if script is run directly
if __name__ == "__main__":
    # Get port from environment variables or use 5001 as default
    port = int(os.environ.get("PORT", 5001))
    # Run app
    app.run(host="0.0.0.0", port=port)
