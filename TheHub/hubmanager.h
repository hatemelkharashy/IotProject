#ifndef HUBMANAGER_H
#define HUBMANAGER_H

#include <QObject>
#include <QMqttClient>
#include <QMqttTopicName>
#include <QByteArray>
#include <QQmlApplicationEngine>
#include <QImageCapture>
#include <QImage>

class HubManager : public QObject
{
    Q_OBJECT
public:
    HubManager(QQmlApplicationEngine *engine = nullptr);

public slots:
    void onMessageReceived(const QByteArray &message, const QMqttTopicName &topic);
    void onSubscriptionMessageReceived(QMqttMessage msg);
    void onConnected();
    void onImageCaptured(int id, const QImage &preview);

private:
    void findCameraObjects(QQmlApplicationEngine *engine);
private:
    QMqttClient *m_client = nullptr;
    QImageCapture *m_imageCapture = nullptr;
};

#endif // HUBMANAGER_H
