# Instructions

Build the image:

    cd docker/
    docker build .

or use the built image in the Zenodo repository

    docker load < docker-image-scaling-d7c0e097.tar.gz

Run the container

    docker run -ti -v $(pwd):/shared -w /shared d7c0e097 mpirun -n 2 python3 demo_plasticity_mohr_coulomb_mpi.py
