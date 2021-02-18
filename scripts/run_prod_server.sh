VERSION=$(cat VERSION)
PORT=8001
echo "RUN mark-yolo5:$VERSION on port $PORT"
docker run -p $PORT:8000 -d mark-yolo5:$VERSION