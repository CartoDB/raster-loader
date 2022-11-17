#!/bin/bash

head -n -3 errors.md > errors.mdx; mv errors.mdx errors.md
head -n -3 README.md > README.mdx; mv README.mdx README.md
head -n -3 utils.md > utils.mdx; mv utils.mdx utils.md
head -n -3 upload.md > upload.mdx; mv upload.mdx upload.md
