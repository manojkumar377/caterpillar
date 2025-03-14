from twilio.rest import Client

# Your Twilio account SID and auth token
account_sid = 'ACd8a95b4c1ddcbb28e6262511b464d621'
auth_token = 'd3e51e59f36ec4fc6d757577a0f523e2'
client = Client(account_sid, auth_token)

# Your Twilio phone number and the destination phone number
twilio_phone_number = '+14707779672'
destination_phone_number = '+918438923377'

# The message you want to send
message_body = 'Hello, this is a test message from your Python script!'

# Sending the message
message = client.messages.create(
    body=message_body,
    from_=twilio_phone_number,
    to=destination_phone_number
)

print(f"Message sent with SID: {message.sid}")
