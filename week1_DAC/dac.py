"""TODO:
generate the label features(vector) with All ConvNet
calculate cosine similarities between images based on the label vectors
select training sample which are ambiguous to classify
train with binary pairwise-classification model

randomly initialize weight

while
    for i in batches:
        sample m size batch
        split training and validation set
        Calculate parameter v
        Update weight by minimizing loss function
    Update lamda(adaptive parameter for controlling the selection)
until l(lamda) > u(lamda)

for all x do
    l := f(x;w)
    c := argmax(l)


loss function ?

lamda -> an adaptive parameter for controlling the selection
u(lamda), l(lamda) -> thresholds ; None -> omitted for training
"""