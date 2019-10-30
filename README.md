
# Hand tracking microservice using Google mediapipe

A microservice based on the hand_tracking_gpu example from mediapipe. Reaches up to 25 fps on a Laptop.

## How to run the micoservice:

* build the docker image from this repository  `docker build https://github.com/davidakr/mediapipe.git`
* run   `docker run -p 9090:9090`
* webserver is running on port 9090

## How to use it:

### Request
base64 encoded image in the request body

### Response
returns a json with following attributes:
* present (boolean): if a hand is detected
* landmarks (int x, int y): landmarks of the hand, if none is detected all are 0
* base64 encoded image (string): the image containing the landmarks and palm detection if available

