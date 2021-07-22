#nsml: nsml/pytorch-mlx:19.10-py3

"""
#nsml: registry.navercorp.com/larva/l2:0.2
#nsml: dsksd/pytorch-apex:19.05-mlx-ling-0.1
#nsml: kprotoss/nvcr-pytorch:19.05-py3-mlx
#nsml: nsml/pytorch-mlx:19.10-py3
#nsml: deepspeed/deepspeed:latest
Copyright 2018 NAVER Corp.
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to
the following conditions:
The above copyright notice and this permission notice shall be included in all copies or substantial
portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


from distutils.core import setup
setup(
    name='nsml BERT reproduce',
    version='1.0',
    description='',
    install_requires = []
    #nsml: floydhub/pytorch:0.3.0-gpu.cuda8cudnn6-py3.17
    #install_requires =['torch==0.4','torchtext==0.2.3','nltk==3.2.5']
)