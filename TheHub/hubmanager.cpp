#include "hubmanager.h"
#include <QMqttClient>
#include <QQmlContext>

HubManager::HubManager(QQmlApplicationEngine *engine)
{
    findCameraObjects(engine);

    m_client = new QMqttClient;
    connect(m_client, &QMqttClient::messageReceived, this, &HubManager::onMessageReceived);
    connect(m_client, &QMqttClient::connected, this, &HubManager::onConnected);
    m_client->setHostname("127.0.0.1");
    m_client->setPort(8883);
    m_client->setClientId("The Hub");
    m_client->connectToHost();;
}

void HubManager::onMessageReceived(const QByteArray &message, const QMqttTopicName &topic)
{
    qDebug() << topic.name();
}

void HubManager::onSubscriptionMessageReceived(QMqttMessage msg)
{
    qDebug() << "Hello";
}

void HubManager::onConnected()
{
    qDebug() << "Connected to broker !";
    auto subscription = m_client->subscribe(QMqttTopicFilter("iotProject/image"));
    if (subscription)
        connect(subscription, &QMqttSubscription::messageReceived, this, &HubManager::onSubscriptionMessageReceived);
}

void HubManager::onImageCaptured(int id, const QImage &preview)
{
    QImage greyScale = preview.convertToFormat(QImage::Format_MonoLSB);
    QImage scaledGreyScale = greyScale.scaled(QSize(64, 64));
    greyScale.save("greyScaleImage.jpeg");
    scaledGreyScale.save("scaledGreyScale.jpeg");
    QByteArray imageBits(scaledGreyScale.bits());
    m_client->publish(QMqttTopicName("iotProject/image"), imageBits);
}

void HubManager::findCameraObjects(QQmlApplicationEngine *engine)
{
    if (!engine)
        return;
    QObject *root = engine->rootObjects().first();
    QImageCapture *imageCapture = root->findChild<QImageCapture *>(QAnyStringView("imageCapture"));
    qDebug() << imageCapture;
    if (imageCapture)
        connect(imageCapture, &QImageCapture::imageCaptured, this, &HubManager::onImageCaptured);
}
