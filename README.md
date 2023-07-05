**Customer Segmentation**
Scenario: Your company now wants to boost its business by focusing on different strategies for different customer groups. Thus, it is important to divide customers into specific groups based on certain factors. Your task is to find those factors and categorize customers into specific groups

You will be provided with a set of data to build a customer segmentation model:

Information about columns in the data file:
* **InvoiceNo**: Invoice Id
* **StockCode**: Ids of purchased items
* **Description**: Description of the item
* **Quantity**: Quantity of the purchased item.
* **InvoiceDate**: Date of purchase
* **UnitPrice**: Price of the item
* **CustomerID**: Id of the customer who made a purchase
* **Country**: Customer's country name


#### 3. RFM model represents customer value:

Since our dataset is limited to sales history and does not include anonymized information about customers, we will use an RFM-based model. The RFM model takes a customer's transactions and computes three important information properties about each customer:
* **Recency**: The value of how recently a customer made a purchase
* **Frequency**: Frequency of customer transactions
* **Monetary value**: The amount (or pounds in ASM post) of all the transactions that the customer has made

![output_30_1](https://github.com/longbui23/CustomerSegmentation/assets/112489957/3bb848f9-c626-4b44-833e-143dd4870410)
![output_36_1](https://github.com/longbui23/CustomerSegmentation/assets/112489957/afb4ad77-f7c9-4860-a75b-8d62248fd457)
![output_43_1](https://github.com/longbui23/CustomerSegmentation/assets/112489957/94b9ea8e-30a4-428c-a35d-eb2ca821d4cf)

Scale RBM on log scale then normalize with StandardScaler
![output_50_2](https://github.com/longbui23/CustomerSegmentation/assets/112489957/87eba649-29ba-4b9f-b244-04d841160ae2)

Regression Visualization
![output_52_0](https://github.com/longbui23/CustomerSegmentation/assets/112489957/6611d827-11d6-4a1c-ad21-68d740bdb9a6)
![output_52_1](https://github.com/longbui23/CustomerSegmentation/assets/112489957/bee07776-e2f3-403e-9825-2c73cceeccc1)

We can see from the graphs above that people who shop with higher frequency (frequency) and more recently visit (recency ) tend to spend more (Monetary value) based on increasing trends. increase of amount with increasing and decreasing trends for Frequency and Recent Hits respectively. .

1) 
+ The relationship between the amount of purchases (amount/Monentary Value) and the recent traffic (recency) is an exponential relationship, that is, the number of rows over time increases exponentially, which proves even more. reasonable business processes, stimulating revenue growth.
+ At the same time, the increase or the exponent of the purchase volume (amount_log) and the increase / exponent of the traffic (recency_log) have a relatively linear relationship, which proves the same thing.

2) 
+ The volume of traffic (frequency) and the quantity of goods purchased (amount/Monentary value) both increase linearly, which means there is no sudden increase in the increase in the class of customers who buy retail/medium/in bulk (penguin). , dolphins, whales) which is steadily increasing in customer classes.

# Modeling - Kmean

#### **Elbow Method:**

Use the Elbow method to find the optimal number of clusters. The idea behind the elbow method is to determine the value of k at which the bend begins to increase fastest. If k increases, the distortion of the clusters will decrease, because the samples will be close to the cluster centroid.

This method considers the percentage of variance explained as a function of the number of clusters. More precisely, if we plot the percentage of variance explained by the clusters against the number of clusters, the first clusters will add a lot of information (which explains more of the variance), but at some point, marginal gain will decrease (Number of clusters increases, variance decreases), creating an angle in the graph, which will be the point of quantity selection for clusters, hence the "elbow criterion". The percentage of variance is explained as the ratio of the group variance to the total variance, also known as the F-test. A slight variation of this method plots the curvature of variance in the cluster

![output_58_0](https://github.com/longbui23/CustomerSegmentation/assets/112489957/5a4b8914-5814-41dc-8fe0-73b7f03d59ed)
![output_58_2](https://github.com/longbui23/CustomerSegmentation/assets/112489957/1434ee2b-0b80-4eaf-9f53-a01a36cb3bc1)
![output_62_0](https://github.com/longbui23/CustomerSegmentation/assets/112489957/eb689452-0c27-4632-8efa-a0555f23df5d)
![output_62_1](https://github.com/longbui23/CustomerSegmentation/assets/112489957/15c90c1f-4dab-4276-bc3a-096366ad27fb)
![output_62_2](https://github.com/longbui23/CustomerSegmentation/assets/112489957/b1e6048c-1f5c-4246-a53c-07c0b7631aa3)

## **Silhouette index: **

**Silhouette** analysis in Kmeans . clustering

Silhouette analysis can be used to study the distance between clusters, as a strategy to quantify the quality of clustering or clustering through visualizations to graph the degree of "group closeness". " of samples in clusters. Silhouette diagrams display a measure of how close each point in a cluster is to points in neighboring clusters and thus provide a way to visually evaluate parameters such as the number of clusters.

This analysis can also be applied to other clustering algorithms besides k-means.

The Silhouette coefficient has a range of [-1, 1], and is calculated by:
1. a(i) is the mean distance between sample x(i) and all other points in the same cluster.
2. b(i) from the next closest cluster is the average distance between sample x(i) and all samples in the nearest cluster.
3. s(i) is the difference between a(i) and b(i) divided by max(a(i), b(i)), as shown here:

\begin{align}
\text{s(i)} = \frac{b(i) - a(i)}{max\{a(i), b(i)\}}
\end{align}

Another way of writing the above formula:

\begin{align}
         \text{s}(i) = \left\{
         \begin{array}{cl}
         1 - a(i)/b(i), & \text{if } a(i) < b(i) \\
         0, & \text{if } a(i) = b(i) \\
         b(i)/a(i) - 1, & \text{if } a(i) > b(i)
         \end{array}
         \right.
     \end{align}

In there:
* If close to +1, it means that the sample is far from neighboring clusters.
* A high value indicates that the object is well matched to its own cluster and poorly matched to neighboring clusters.
* If most of the objects have high values, then the clustering configuration is appropriate.
* If many points have low or negative values, the clustering configuration may have too many or too few clusters.
* A value of 0 indicates that the sample is on or very close to the decision boundary between two neighboring clusters
* Negative values indicate that those samples may have been assigned to the wrong group.

K is considered bad when clusters have below-average Silhouette scores or there is large variation in the size of Silhouette cells. K is considered good when all cells have similar or not much different thickness, or in other words all cells have similar dimensions.

Although we must be aware that in some cases and situations we may sometimes have to discard the mathematical interpretation given by the algorithm and consider the business relevance of the result obtained.

Let's see how our data behaves for each of the K values (3, 5 and 7) in the Silhouette score of each cluster, along with the centroid of each cluster highlighted in the scatter plots, they We will cluster with 3 input variables amount_log, recency_log, frequency_log but will display on scatter in pairs (amount_log, recency_log) and (amount_log, frequency_log)

## 3 clusters: 
![output_76_0](https://github.com/longbui23/CustomerSegmentation/assets/112489957/70b21c75-3ec6-4706-9bf2-7af03f944418)

## 5 clusters:
![output_79_0](https://github.com/longbui23/CustomerSegmentation/assets/112489957/952cc400-e702-4499-815e-e6a08270472e)

##7 clusters:
![output_82_0](https://github.com/longbui23/CustomerSegmentation/assets/112489957/c631e465-8e01-4fd4-a083-0c6a74d24584)

## Groups of customers by methods of clustering: 
![output_86_0](https://github.com/longbui23/CustomerSegmentation/assets/112489957/61921dd0-c6a8-41f7-8bc8-cf838714f86d)

## Change of amount by different groups of clusterings:
![output_89_0](https://github.com/longbui23/CustomerSegmentation/assets/112489957/5d1a77cd-7cf1-4798-ab4b-d658e4a95192)
![output_90_0](https://github.com/longbui23/CustomerSegmentation/assets/112489957/4062370a-e3a6-42df-9ca5-89e4efc2ac73)
![output_91_0](https://github.com/longbui23/CustomerSegmentation/assets/112489957/d7f65283-c678-4990-bf4a-088064b9509d)
