library(arules)
library(dplyr)
library(arulesViz)
library(networkD3)
library(visNetwork)
library(rlang)
library(usethis)
library(devtools)
library(base64enc)
library(RCurl)
library(httr)
library(twitteR)
library(htmlwidgets)


twitteR::setup_twitter_oauth(consumer_key='qYMHJsgccjkgyMBOJxSYHqnxF',
                             consumer_secret='nWVpvqtBY1mLRR5CU3vYV3rpMuX5XE23oKhPkaCDeZsWshLTT1',
                             access_token='951788659849576448-EIGGXXZdnd0GRnzCdS5hDEANzQ9mme8',
                             access_secret='85m7x1xCbSMN7euFFxu3ePuOEgDlwCHujKb1uUlLQEpSY')

Search<- twitteR::searchTwitter("Fortune 500 CEO", n=1000, lang="en")
Search_DF <- twitteR::twListToDF(Search)
TransactionTweetsFile = "TweetResults.csv"
Trans <- file(TransactionTweetsFile)


Tokens<- tokenizers::tokenize_tweets(Search_DF$text[1],stopwords = stopwords::stopwords("en"), 
              lowercase = TRUE,  strip_punct = TRUE, 
              strip_url = TRUE, simplify = TRUE)
cat(unlist(Tokens), "\n", file=Trans, sep=",")
close(Trans)

## Append remaining lists of tokens into file
Trans <- file(TransactionTweetsFile, open = "a")
for(i in 2:nrow(Search_DF)){
  Tokens<-tokenizers::tokenize_tweets(Search_DF$text[i],stopwords = stopwords::stopwords("en"), 
                                      lowercase = TRUE,  strip_punct = TRUE, 
                                      strip_url = TRUE, simplify = TRUE)
  
  cat(unlist(Tokens), "\n", file=Trans, sep=",")
  cat(unlist(Tokens))
}

close(Trans)


# read in dataframe
TweetDF <- read.csv(TransactionTweetsFile, 
                    header = FALSE, sep = ",")


TweetDF<-TweetDF %>%
  mutate_all(as.character)

# We can now remove certain words
TweetDF[TweetDF == "t.co"] <- ""
TweetDF[TweetDF == "rt"] <- ""
TweetDF[TweetDF == "http"] <- ""
TweetDF[TweetDF == "https"] <- ""
TweetDF[TweetDF == "named"] <- ""
TweetDF[TweetDF == "mylan"] <- ""
TweetDF[TweetDF == "become"] <- ""

for(i in 1:14){
  for (k in 1:1000){
  if (grepl('@', TweetDF[[i]][k])==TRUE){
    TweetDF[TweetDF==TweetDF[[i]][k]]<- ""
  }
  }
}
for(i in 1:14){
  for (k in 1:1000){
    if (grepl('#', TweetDF[[i]][k])==TRUE){
      TweetDF[TweetDF==TweetDF[[i]][k]]<- ""
    }
  }
}




## Check it so far....
TweetDF

## Clean with grepl - every row in each column
MyDF<-NULL
MyDF2<-NULL
for (i in 1:ncol(TweetDF)){
  MyList=c() 
  MyList2=c() # each list is a column of logicals ...
  MyList=c(MyList,grepl("[[:digit:]]", TweetDF[[i]]))
  MyDF<-cbind(MyDF,MyList)  ## create a logical DF
  MyList2=c(MyList2,(nchar(TweetDF[[i]])<4 | nchar(TweetDF[[i]])>11))
  MyDF2<-cbind(MyDF2,MyList2) 
  ## TRUE is when a cell has a word that contains digits
}
## For all TRUE, replace with blank
TweetDF[MyDF] <- ""
TweetDF[MyDF2] <- ""
(head(TweetDF,10))

# Now we save the dataframe using the write table command 
write.table(TweetDF, file = "UpdatedTweetFile.csv", col.names = FALSE, 
            row.names = FALSE, sep = ",")
TweetTrans <- read.transactions("UpdatedTweetFile.csv", sep =",", 
                                format("basket"),  rm.duplicates = TRUE)



############ Create the Rules  - Relationships ###########
TweetTrans_rules = arules::apriori(TweetTrans, 
                                   parameter = list(support=.5, conf=0.7, minlen=1))


##  SOrt by Conf
SortedRules_conf <- sort(TweetTrans_rules, by="confidence", decreasing=TRUE)
inspect(SortedRules_conf[1:15])
## Sort by Sup
SortedRules_sup <- sort(TweetTrans_rules, by="support", decreasing=TRUE)
inspect(SortedRules_sup[1:15])
## Sort by Lift
SortedRules_lift <- sort(TweetTrans_rules, by="lift", decreasing=TRUE)
inspect(SortedRules_lift[1:15])





## Convert the RULES to a DATAFRAME
Rules_DF2<-DATAFRAME(TweetTrans_rules, separate = TRUE)
(head(Rules_DF2))

## Convert to char
Rules_DF2$LHS<-as.character(Rules_DF2$LHS)
Rules_DF2$RHS<-as.character(Rules_DF2$RHS)

## Remove all {}
Rules_DF2[] <- lapply(Rules_DF2, gsub, pattern='[{]', replacement='')
Rules_DF2[] <- lapply(Rules_DF2, gsub, pattern='[}]', replacement='')

Rules_DF2




###########################################
###### Do for SUp, Conf, and Lift   #######
###########################################
## Remove the sup, conf, and count
## USING LIFT
Rules_L<-Rules_DF2[c(1,2,5)]
names(Rules_L) <- c("SourceName", "TargetName", "Weight")
head(Rules_L,30)

## USING SUP
Rules_S<-Rules_DF2[c(1,2,3)]
names(Rules_S) <- c("SourceName", "TargetName", "Weight")
head(Rules_S,30)

## USING CONF
Rules_C<-Rules_DF2[c(1,2,4)]
names(Rules_C) <- c("SourceName", "TargetName", "Weight")
head(Rules_C,30)

## CHoose and set  (Lift is chosen)

#Rules_Sup<-Rules_C
Rules_Sup<-Rules_L
#Rules_Sup<-Rules_S


###########################################################################
#############       Build a NetworkD3 edgeList and nodeList    ############
###########################################################################

#edgeList<-Rules_Sup
# Create a graph. Use simplyfy to ensure that there are no duplicated edges or self loops
#MyGraph <- igraph::simplify(igraph::graph.data.frame(edgeList, directed=TRUE))
#plot(MyGraph)

############################### BUILD THE NODES & EDGES ####################################
(edgeList<-Rules_Sup)
(MyGraph <- igraph::simplify(igraph::graph.data.frame(edgeList, directed=TRUE)))

nodeList <- data.frame(ID = c(0:(igraph::vcount(MyGraph) - 1)), 
                       # because networkD3 library requires IDs to start at 0
                       nName = igraph::V(MyGraph)$name)
## Node Degree
(nodeList <- cbind(nodeList, nodeDegree=igraph::degree(MyGraph, 
                                                       v = igraph::V(MyGraph), mode = "all")))

## Betweenness
BetweenNess <- igraph::betweenness(MyGraph, 
                                   v = igraph::V(MyGraph), 
                                   directed = TRUE) 

(nodeList <- cbind(nodeList, nodeBetweenness=BetweenNess))







###################################################################################
########## BUILD THE EDGES #####################################################
#############################################################
# Recall that ... 
# edgeList<-Rules_Sup
getNodeID <- function(x){
  which(x == igraph::V(MyGraph)$name) - 1  #IDs start at 0
}


edgeList <- plyr::ddply(
  Rules_Sup, .variables = c("SourceName", "TargetName" , "Weight"), 
  function (x) data.frame(SourceID = getNodeID(x$SourceName), 
                          TargetID = getNodeID(x$TargetName)))

head(edgeList)
nrow(edgeList)


########################################################################
##############  Dice Sim ################################################
###########################################################################
#Calculate Dice similarities between all pairs of nodes
#The Dice similarity coefficient of two vertices is twice 
#the number of common neighbors divided by the sum of the degrees 
#of the vertices. Method dice calculates the pairwise Dice similarities 
#for some (or all) of the vertices. 
DiceSim <- igraph::similarity.dice(MyGraph, vids = igraph::V(MyGraph), mode = "all")
head(DiceSim)

#Create  data frame that contains the Dice similarity between any two vertices
F1 <- function(x) {data.frame(diceSim = DiceSim[x$SourceID +1, x$TargetID + 1])}
#Place a new column in edgeList with the Dice Sim
head(edgeList)
edgeList <- plyr::ddply(edgeList,
                        .variables=c("SourceName", "TargetName", "Weight", 
                                     "SourceID", "TargetID"), 
                        function(x) data.frame(F1(x)))
head(edgeList)

##################################################################################
##################   color #################################################
######################################################
# COLOR_P <- colorRampPalette(c("#00FF00", "#FF0000"), 
#                             bias = nrow(edgeList), space = "rgb", 
#                             interpolate = "linear")
# COLOR_P
# (colCodes <- COLOR_P(length(unique(edgeList$diceSim))))
# edges_col <- sapply(edgeList$diceSim, 
#                     function(x) colCodes[which(sort(unique(edgeList$diceSim)) == x)])
# nrow(edges_col)




#######################################################
########  NetworkD3 plot   ###########
#######################################################

D3_network_Tweets <- networkD3::forceNetwork(
  Links = edgeList, # data frame that contains info about edges
  Nodes = nodeList, # data frame that contains info about nodes
  Source = "SourceID", # ID of source node 
  Target = "TargetID", # ID of target node
  Value = "Weight", # value from the edge list (data frame) that will be used to value/weight relationship amongst nodes
  NodeID = "nName", # value from the node list (data frame) that contains node description we want to use (e.g., node name)
  Nodesize = "nodeBetweenness",  # value from the node list (data frame) that contains value we want to use for a node size
  Group = "nodeDegree",  # value from the node list (data frame) that contains value we want to use for node color
  height = 700, # Size of the plot (vertical)
  width = 900,  # Size of the plot (horizontal)
  fontSize = 20, # Font size
  linkDistance = networkD3::JS("function(d) { return d.value*1000; }"), # Function to determine distance between any two nodes, uses variables already defined in forceNetwork function (not variables from a data frame)
  linkWidth = networkD3::JS("function(d) { return d.value*5; }"),# Function to determine link/edge thickness, uses variables already defined in forceNetwork function (not variables from a data frame)
  opacity = 5, # opacity
  zoom = TRUE, # ability to zoom when click on the node
  opacityNoHover = 5, # opacity of labels when static
  linkColour = "red"   ###"edges_col"red"# edge colors
) 

# Plot network
D3_network_Tweets

# Save network as html file
networkD3::saveNetwork(D3_network_Tweets, 
                       "networkD3.html", selfcontained = TRUE)



#######################################################
########  Sankey plot   ###########
#######################################################
sankey<-networkD3::sankeyNetwork(Links = edgeList, 
                         Nodes = nodeList, 
                         Source = "SourceID",
                         Target = "TargetID", 
                         Value = "Weight", 
                         NodeID = "nName")

sankey

# Save as html file
networkD3::saveNetwork(sankey, 
                       "sankey.html", selfcontained = TRUE)


#######################################################
########  VisNetwork plot   ###########
#######################################################

Myedges<-edgeList[c(4,5,3)]
Mynodes<-nodeList[c(1,2)]
colnames(Myedges)<-c('from','to','weight')
colnames(Mynodes)<-c('id','label')
Myedges$weight<-as.numeric(Myedges$weight)


## plot
(nodes<-Mynodes)
edges<-Myedges
head(edges)

curve<-visNetwork(nodes, edges, layout = "layout_with_fr",
           arrows="middle")
curve

edges
(edges <- mutate(edges, width = weight*10, length=40))


line<-visNetwork(nodes, edges) %>% 
  visIgraphLayout(layout = "layout_with_fr") %>% 
  visEdges(arrows = "middle")
line

## Save as html file
saveWidget(curve, file = "visNet 1.html")
saveWidget(line, file = "visNet 2.html")


