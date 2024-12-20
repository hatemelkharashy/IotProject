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
    Q_PROPERTY(bool connected READ connected WRITE setConnected NOTIFY connectedChanged)
    Q_PROPERTY(QString gesture READ gesture WRITE setGesture NOTIFY gestureChanged FINAL)
public:
    HubManager(QQmlApplicationEngine *engine = nullptr);

    QString gesture() const;
    void setGesture(const QString &newGesture);

public slots:
    void onMessageReceived(const QByteArray &message, const QMqttTopicName &topic);
    void onSubscriptionMessageReceived(QMqttMessage msg);
    void onConnected();
    void onImageCaptured(int id, const QImage &preview);

    void setConnected(bool val);
    bool connected() const;
    void findCameraObjects(QQmlApplicationEngine *engine);

signals:
    void connectedChanged();

    void gestureChanged();

private:
    QMqttClient *m_client = nullptr;
    QImageCapture *m_imageCapture = nullptr;
    bool m_connected = false;
    QString m_gesture = QStringLiteral("No Detection");
};

#endif // HUBMANAGER_H
