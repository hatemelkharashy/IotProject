import QtQuick
import QtQuick.Window
import QtMultimedia
import QtQuick.Controls
import QtQuick.Layouts

Window {
    width: 480
    height: 800
    visible: true
    title: qsTr("Hello World")

    MediaDevices {
        id: mediaDevices
    }
    CaptureSession {
        imageCapture: ImageCapture {
            objectName: "imageCapture"
            id: imageCapture
        }

        camera: Camera {
            cameraDevice: mediaDevices.defaultVideoInput
            active: true
            onCameraDeviceChanged: {
                console.log(mediaDevices.defaultVideoInput)
            }
        }
        videoOutput: videoOutput
    }

    ColumnLayout {
        Layout.preferredWidth: parent.width
        Layout.preferredHeight: parent.height
        spacing: 10
        Button {
            Layout.preferredWidth: 150
            Layout.preferredHeight: 50
            Layout.alignment: Qt.AlignHCenter
            text: "Capture"
            onClicked: {
                imageCapture.capture()
            }
        }

        Item {
            id: brokerConnection
            Layout.preferredWidth: parent.width
            Layout.preferredHeight: 40
            Layout.alignment: Qt.AlignHCenter
            Row {
                anchors.centerIn: parent
                spacing: 15
                Rectangle {
                    width: 20
                    height: 20
                    radius: 10
                    color: HubManager.connected ? "green" : "red"
                }
                Text {
                    text: HubManager.connected ? "Connected" : "Not Connected"
                }
            }
        }

        Item {
            id: gestureTopic
            Layout.preferredWidth: parent.width
            Layout.preferredHeight: 40
            Layout.alignment: Qt.AlignHCenter
            Row {
                anchors.centerIn: parent
                spacing: 15
                Text {
                    text: "Gesture Received : "
                }
                Text {
                    text: HubManager.gesture
                }
            }
        }

        VideoOutput {
            id: videoOutput
            Layout.preferredWidth: 480
            Layout.preferredHeight: 600
            Layout.maximumWidth: 480
            Layout.maximumHeight: 600
            Layout.fillWidth: true
            Layout.fillHeight: true
        }
    }
}
