# Next_aso-


Problem statement - There are times when a user writes Good, Nice App or any other positive text, in the review and gives 1-star rating. Your goal is to identify the reviews where the semantics of review text does not match rating. 

Your goal is to identify such ratings where review text is good, but rating is negative- so that the support team can point this to users. 

Deploy it using - Flask/Streamlit etc and share the live link. 

Important
In this data app - the user will upload a csv and you would be required to display the reviews where the content doesn’t match ratings.  This csv will be in the same format as the DataSet Link

Github link: (https://share.streamlit.io/harshadap30/next_aso-/main/Crome_review.py)


Question 3: Ranking Data - Understanding the co-relation between keyword rankings with description or any other attribute.
Answer: Owing to shortage of time, I was unable to develop a functioning model which would experimentally answer these questions. I will make an attempt to answer it here theoretically.

Ranking has direct correlation to the keyword used in search and presence of that keyword in either the app_id directly or in the app description.
Early presence of keyword will impact the ranking as even with humans we see that people tend to look for catch words in the initial couple of sentences.
APP ID has direct impact on the ranking as the search keyword, if present in the app_id itself, will impact in improving the ranking of the app in playstore.
Another parameter that would affect the ranking is how many times an user who is looking out for a particular functionality chooses to press the app link. That is determined by the type of catchy adjectives used to explain about the in the short description. Like easy to use, free to use, etc.
Short description will be more catchy if they are precise, than long descriptions

Q1. Write about any difficult problem that you solved. (According to us difficult - is something which 90% of people would have only 10% probability in getting a similarly good solution). 
Ans : The difficult problem I have solved is to find no of layers for deep neual network model.As I was working on a project where I need to create a classfication deep neural network model which can classfy the position(Y) of a player depending upon the stats (X) of  a player
with trial and errors of learing rate and epochs and different neural layers of a model I came to the solution with new hyperparamters 




Q2. Ordered pairs of real numbers (a,b) a,b∈R form a vector space V. Which of the following is a subspace of V?
•	The set of pairs (a, a + 1) for all real a
•	The set of pairs (a, b) for all real a ≥ b
•	The set of pairs (a, 2a) for all real a
•	The set of pairs (a, b) for all non-negative real a,b
Ans: For a subspace H to be a vector space by itself, it has to fulfil the following 3 criteria.
(i)	Zero - The zero vector of vector space V must also be in H
(ii)	Addition - For each u,v in H, u+v is also in H.
(iii)	Scalar multiplication - For each u in H and a scalar c, cu is also in H.
Let us consider each of the above sets and discuss whether they form a subspace of V
•	The set of pairs (a, a + 1) for all real a

clearly (0,0) which is the zero of vector space V is not in this set. So it can’t be a subspace

•	The set of pairs (a, b) for all real a ≥ b

It meets the first two criteria discussed above. It has the zero vector (0,0) in it. It also satisfies the addition rule. But it fails to meet the scalar multiplication rule. Ex: let c = -5, u = (4,3) cu = (-20,-15) which is not an element of the given set.

•	The set of pairs (a, 2a) for all real a

It meets all the 3 criteria. It has the zero vector (0,0) in it. It meets the addition rule. Ex: for any u = (a,2a) and v = (b,2b), u+v = (a+b, 2(a+b)) which belongs to the given set. It also meets the scalar multiplication criterion. For any u = (a,2a), cu = (ca, 2ca) which belongs to the given set. So, the set of pairs (a,2a) for all real a, form the subspace of vector space V.

•	The set of pairs (a, b) for all non-negative real a,b

It satisfies the first two criteria for being a subspace as it contains both zero vector and satisfies the addition rule for all the elements in the set. But it fails to meet the scalar multiplication criterion, for a choice of negative value for scalar c. ex: if c = -1, an element of the set u = (3,4) , cu = (-3,-4) which is not a member of the given set.
