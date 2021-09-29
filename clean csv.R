fortune1000<-read.csv(file="/Users/balogZ/Desktop/501 Assignment 1/Fortune_1000.csv")
df<- data.frame(fortune1000)


# Only keep the first co-CEO if multiple co-CEO exist (for the wikipedia api to work)

ceo_names<- df$CEO
for(i in 1:1000){
  ceo_names[i]<- gsub("/.*","",ceo_names[i])
}
df$CEO<-ceo_names


# Drop columns that are irrelevant 

df<-df[, !(colnames(df) %in% c("state","rank_change","revenue", "profit", "num..of.employees", 
                               "city", "prev_rank", "Website", "Ticker", "Market.Cap", "profitable"))]


# Reorder the columns 

df<-df[, c("CEO", "company" ,"rank"  , "sector"  , "newcomer" , "ceo_founder", "ceo_woman" )]


# Write to new csv file

write.csv(df, "/Users/balogZ/Desktop/501 Assignment 2/R cleaned.csv")
