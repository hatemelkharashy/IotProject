import time
import mqtt_client as MqttClient
import random

my_mqtt_topic_hello = "me/feeds/hello"  # the topic we send on
image_topic = "iotProject/image"  # Rpi pico receives image on this topic

client = MqttClient.MqttClient()
client.subscribe(image_topic)
last_msg_send_time = 0

while True:
    client.loop()
    if time.monotonic() - last_msg_send_time > 3.0:  # send a message every 3 secs
        client.publish(my_mqtt_topic_hello, "Hello World")