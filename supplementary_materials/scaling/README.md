# Instructions

Build the image:

    cd docker/
    docker build .

or use the built image in the Zenodo repository

    docker load < docker-image-scaling-e48acc59.tar.gz

Run the container

    docker run -ti -v $(pwd):/shared -w /shared e48acc59 mpirun -n 2 python3 demo_plasticity_mohr_coulomb_mpi.py
