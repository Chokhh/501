library("selectr")
library("rvest")
library("xml2")
library("jsonlite")
library(httr)


# Write a loop that gather text information based on each of the CEO names

fortune1000<-read.csv(file="Fortune_1000.csv")
ceo_names<- fortune1000$CEO
for(i in 1:1000){
  name<-URLencode(ceo_names[i]) # the space has to be replace by "%20"
  api<- "https://en.wikipedia.org/w/api.php?"
  request_url<- paste(api,"action=query&format=json&prop=extracts&titles=",name,"&formatversion=2",sep="" )
  wikipage_text <- html_text(read_html(request_url))
  write(wikipage_text, "wikipage.txt", append=TRUE)
}



