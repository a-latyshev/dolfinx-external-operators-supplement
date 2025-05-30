# Instructions

Build the image:

    cd docker/
    docker build .

or use the built image in the Zenodo repository

    docker load < docker-image-plots-ebdbd829.tar.gz

Run the container

    docker run -ti -v $(pwd):/shared -w /shared ebdbd829 python3 plots_for_paper.py
