import os
import time
import ssl, socketpool, wifi
import adafruit_minimqtt.adafruit_minimqtt as MQTT

class MqttClient:
    def __init__(self):
        self.__connectToWiFi()
        self.__createClient()

    def __connectToWiFi(self):
        # Connect to WiFi
        print(f"Connecting to {os.getenv('CIRCUITPY_WIFI_SSID')}")
        wifi.radio.connect(os.getenv("CIRCUITPY_WIFI_SSID"), os.getenv("CIRCUITPY_WIFI_PASSWORD"))

    def __createClient(self):
        mqtt_client = MQTT.MQTT(
            broker=os.getenv("mqtt_broker"),
            port=os.getenv("mqtt_port"),
            username=os.getenv("mqtt_username"),
            password=os.getenv("mqtt_password"),
            socket_pool=socketpool.SocketPool(wifi.radio),
            ssl_context=ssl.create_default_context(),
        )
        mqtt_client.on_connect = self.connected
        mqtt_client.on_disconnect = self.disconnected
        mqtt_client.on_message = self.message
        print("Connecting to MQTT broker...")
        print(f"Connecting to {os.getenv('mqtt_broker')}")
        print(f"Connecting to {os.getenv('mqtt_port')}")
        mqtt_client.connect()
        self.client = mqtt_client

    def connected(self, client, userdata, flags, rc):
        print("Connected to MQTT Broker!")

    def disconnected(self, client, userdata, rc):
        print("Disconnected from MQTT Broker!")

    def message(self, client, topic, message):
        print("New message on topic {0}: {1}".format(topic, message))
        val = 0
        try: 
            val = int(message)  # attempt to parse it as a number
        except ValueError:
            pass
        print("setting LED to color:",val)

    def subscribe(self, topic):
        self.client.subscribe(topic)
        
    def publish(self, topic, msg):
        measuredTime = time.monotonic()
        self.client.publish(topic, msg)
        print("Time taken to publish a message : ", str(time.monotonic() - measuredTime))
        
    def loop(self):
        self.client.loop(timeout=1)    