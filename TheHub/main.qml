import QtQuick
import QtQuick.Window
import QtMultimedia
import QtQuick.Controls
Window {
    width: 640
    height: 1920
    visible: true
    title: qsTr("Hello World")

    Button {
        width: 200
        height: 200
        text: "Capture"
        onClicked: {
            console.log("Capturing")
            imageCapture.capture()
            previewImage.source = imageCapture.preview
        }
    }

    MediaDevices {
        id: mediaDevices
        onVideoInputsChanged: {
            console.log("Hello")
        }
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
    VideoOutput {
        id: videoOutput
        width: 640
        height: 480
    }
    Image {
        id: previewImage
        y: 490
    }

    Image {
        objectName: "greyScaleImage"
        id: greyScaleImage
        y: 1000
    }
}
