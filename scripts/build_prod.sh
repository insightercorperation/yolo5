VERSION=$(cat VERSION)
echo "Build mark-yolo5:$VERSION image"
docker build . -f Dockerfile -t mark-yolo5:$VERSION