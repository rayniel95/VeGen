# About

A [generalized vector model][1] implementation. Implemented an algorithm for [frequency matrix creation][2], [tf idf matrix][3] creation, different similarities ([cosine][4], [jaccard][5], etc.) and a `vegen` method that transform tf idf matrix and the vector consult to the generalized vector model space.

[1]: https://en.wikipedia.org/wiki/Generalized_vector_space_model
[2]: https://en.wikipedia.org/wiki/Document-term_matrix
[3]: https://en.wikipedia.org/wiki/Tf%E2%80%93idf
[4]: https://en.wikipedia.org/wiki/Cosine_similarity
[5]: https://en.wikipedia.org/wiki/Jaccard_index

## More

See in this repository:

- `6.pdf`
- `general-vsm.pdf`
- `E09-3009.pdf`

# Requirements

- Docker

# How to execute

1. Build image `sudo docker build ./ -t rayniel95/vegen:v1.0`
2. Execute container `sudo docker run -it rayniel95/vegen:v1.0` 

# How to use

Call method `vegen` at `./code/vegen.py`.
