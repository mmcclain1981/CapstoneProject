knitr::opts_chunk$set(echo = TRUE)

library(ggplot2)
library(tidytext)
library(dplyr)
library(stringr)
library(lubridate)
library(forcats)
library(wordcloud)
library(textdata)
library(factoextra)
library(topicmodels)
library(formattable)
library(scales)
library(caTools)
library(randomForest)

# Read the contents of the csv file into a dataframe and explore the data
chat <- read.csv("counsel_chat2.csv")
summary(chat)

# Place questions into a separate frame, count them by topic and visualize
questions <- data.frame(chat['questionID'],chat['questionTitle'],chat['questionText'],chat['topic'])
questions <- unique(questions)
summary(questions)
ggplot(questions,aes(y = fct_rev(fct_infreq(topic)))) +
  geom_bar(fill = 'navy') +
  labs(title = "Question Counts by Topic", x = "Counts", y = "Topics")

# Rate therapists by upvotes overall; visualize
chat$therapist <- substring(chat$therapistInfo,1,20)
therapists <- chat %>%
  group_by(therapist) %>%
  summarize(upvotes = sum(upvotes)) %>%
  slice_max(n=25, order_by = upvotes)
ggplot(therapists,aes(y = fct_reorder(therapist,upvotes), x = upvotes)) +
  geom_col(fill = 'navy') +
  labs(title = "Upvotes by Therapist", x = "Upvotes", y = "Therapists")

# Therapist(s) with maximum upvotes within each topic
topic_therapist <- chat %>%
  group_by(topic, therapist) %>%
  summarize(upvotes = sum(upvotes)) %>%
  filter(upvotes == max(upvotes))
topic_therapist

# ***** Sentiment Analysis *****
# For reference on how to use tidytext library: Silge and Robinson (2016)

# Prepare the dataframe of questions for sentiment analysis
data(stop_words)
tidy_q <- questions %>%
  unnest_tokens(word, questionText) %>%
  anti_join(stop_words) %>%
  filter(str_detect(word,"[a-z]"),
         !word == "feel",
         !word == "feeling",
         !word == "feelings",
         !word == "i'm",
         !word == "time") %>%
  mutate(word = replace(word, word == "depressed", "depression"),
         word = replace(word, word == "relationships", "relationship"))

# Get summarized sentiment scores from AFINN lexicon; visualize
afinn_q <- tidy_q %>% 
  inner_join(get_sentiments("afinn")) %>% 
  group_by(topic) %>% 
  summarise(sentiment = sum(value)) %>% 
  mutate(method = "AFINN")
ggplot(afinn_q, aes(y = fct_rev(fct_reorder(topic,sentiment)), x = sentiment)) +
  geom_col(show.legend = FALSE, fill = 'navy') +
  labs(title = "Summarized Sentiment Scores by Topic", x = "Summarized Scores", y = "Topics")

# Get average sentiment scores by topic from AFINN lexicon; visualize
afinn_t_avg <- tidy_q %>% 
  inner_join(get_sentiments("afinn")) %>% 
  group_by(topic) %>% 
  summarise(sentiment = mean(value))
ggplot(afinn_t_avg, aes(y = fct_rev(fct_reorder(topic,sentiment)), x = sentiment)) +
  geom_col(show.legend = FALSE, fill = 'navy') +
  labs(title = "Average Sentiment Scores by Topic", x = "Average Scores", y = "Topics")

# Get sentiment coefficients from NRC lexicon
nrc_t <- tidy_q %>%
  inner_join(get_sentiments("nrc")) %>%
  group_by(topic) %>%
  summarise(ttl_words = n(),
            anger_coef = sum(sentiment == "anger") / ttl_words,
            fear_coef = sum(sentiment == "fear") / ttl_words,
            sad_coef = sum(sentiment == "sadness") / ttl_words,
            disgust_coef = sum(sentiment == "disgust") / ttl_words)

# Get sentiment scores from AFINN by question
afinn_q_avg <- tidy_q %>% 
  inner_join(get_sentiments("afinn")) %>% 
  group_by(questionID) %>% 
  summarise(sentiment_score = mean(value))

# Get NRC sentiment scores by question
nrc_q_anger <- tidy_q %>%
  inner_join(get_sentiments("nrc")) %>%
  filter(sentiment == "anger") %>%
  group_by(questionID) %>%
  summarise(anger_cnt = n())
nrc_q_fear <- tidy_q %>%
  inner_join(get_sentiments("nrc")) %>%
  filter(sentiment == "fear") %>%
  group_by(questionID) %>%
  summarise(fear_cnt = n())
nrc_q_sad <- tidy_q %>%
  inner_join(get_sentiments("nrc")) %>%
  filter(sentiment == "sadness") %>%
  group_by(questionID) %>%
  summarise(sad_cnt = n())
nrc_q_disgust <- tidy_q %>%
  inner_join(get_sentiments("nrc")) %>%
  filter(sentiment == "disgust") %>%
  group_by(questionID) %>%
  summarise(disgust_cnt = n())
q_stats <- tidy_q %>%
  group_by(questionID) %>%
  summarise(ttl_words = n())

# Compose the list of absolutist words
abs_words <- c("absolutely", "all", "always", "complete", "completely", "constant", "constantly",
               "definitely", "entire", "entirely", "ever", "every", "everyone", "everything",
               "full", "must", "never", "nothing", "totally", "whole")

# Count absolutist words by question
abs_q_count <- tidy_q %>%
  filter(word %in% abs_words) %>%
  group_by(questionID) %>%
  summarise(abs_cnt = n())

# Combine sentiment scores and absolutist words counts into one dataframe
q_stats <- merge(q_stats, nrc_q_anger, by = "questionID", all = TRUE)
q_stats <- merge(q_stats, nrc_q_fear, by = "questionID", all = TRUE)
q_stats <- merge(q_stats, nrc_q_sad, by = "questionID", all = TRUE)
q_stats <- merge(q_stats, nrc_q_disgust, by = "questionID", all = TRUE)
q_stats <- merge(q_stats, afinn_q_avg, by = "questionID", all = TRUE)
q_stats <- merge(q_stats, abs_q_count, by = "questionID", all = TRUE)

# Fix the missing data
q_stats[is.na(q_stats)] <- 0

# Calculate coefficients based on counts
q_stats <- q_stats %>%
  mutate(anger_coef = anger_cnt /ttl_words,
         fear_coef = fear_cnt / ttl_words,
         sad_coef = sad_cnt / ttl_words,
         disgust_coef = disgust_cnt / ttl_words,
         abs_coef = abs_cnt / ttl_words)

# Keep only coefficients, remove counts
q_stats <- q_stats %>%
  select(-c(ttl_words,anger_cnt,fear_cnt,sad_cnt,disgust_cnt,abs_cnt))

# Remove the question ID column - not needed
tmp_stats <- q_stats[-1]

# ***** Enhanced Clustering Analysis *****
# The article "Cluster Analysis in R Simplified and Enhanced" retrieved from datanovia.com
# was used for reference

# Scale the dataset, create and visualize the dissimilarity matrix
q_sstats <- scale(tmp_stats)
q_dist <- get_dist(q_sstats, method = "pearson")
fviz_dist(q_dist)

# Perform ehnanced clustering
q_clust <- eclust(q_sstats,"kmeans",iter.max = 30, nstart = 25)

# Show the optimal number of clusters
fviz_gap_stat(q_clust$gap_stat)

# Select top 4 topics from the original dataset of questions with most available data
sel_topics <- c("depression","relationships","intimacy","anxiety")
sel_questions <- questions %>%
  filter(topic %in% sel_topics)

# Repeat the same steps outlined above on the new dataset of four topics
tidy_sq <- sel_questions %>%
  unnest_tokens(word, questionText) %>%
  anti_join(stop_words) %>%
  filter(str_detect(word,"[a-z]"),
         !word == "i'm") %>%
  mutate(word = replace(word, word == "depressed", "depression"),
         word = replace(word, word == "relationships", "relationship"))
afinn_sq_avg <- tidy_sq %>% 
  inner_join(get_sentiments("afinn")) %>% 
  group_by(questionID) %>% 
  summarise(sentiment_score = mean(value))
nrc_sq_anger <- tidy_sq %>%
  inner_join(get_sentiments("nrc")) %>%
  filter(sentiment == "anger") %>%
  group_by(questionID) %>%
  summarise(anger_cnt = n())
nrc_sq_fear <- tidy_sq %>%
  inner_join(get_sentiments("nrc")) %>%
  filter(sentiment == "fear") %>%
  group_by(questionID) %>%
  summarise(fear_cnt = n())
nrc_sq_sad <- tidy_sq %>%
  inner_join(get_sentiments("nrc")) %>%
  filter(sentiment == "sadness") %>%
  group_by(questionID) %>%
  summarise(sad_cnt = n())
nrc_sq_disgust <- tidy_sq %>%
  inner_join(get_sentiments("nrc")) %>%
  filter(sentiment == "disgust") %>%
  group_by(questionID) %>%
  summarise(disgust_cnt = n())
nrc_sq_joy <- tidy_sq %>%
  inner_join(get_sentiments("nrc")) %>%
  filter(sentiment == "joy") %>%
  group_by(questionID) %>%
  summarise(joy_cnt = n())
nrc_sq_trust <- tidy_sq %>%
  inner_join(get_sentiments("nrc")) %>%
  filter(sentiment == "trust") %>%
  group_by(questionID) %>%
  summarise(trust_cnt = n())
abs_sq_count <- tidy_sq %>%
  filter(word %in% abs_words) %>%
  group_by(questionID) %>%
  summarise(abs_cnt = n())
sq_stats <- tidy_sq %>%
  group_by(questionID) %>%
  summarise(ttl_words = n())
sq_stats <- merge(sq_stats, nrc_sq_anger, by = "questionID", all = TRUE)
sq_stats <- merge(sq_stats, nrc_sq_fear, by = "questionID", all = TRUE)
sq_stats <- merge(sq_stats, nrc_sq_sad, by = "questionID", all = TRUE)
sq_stats <- merge(sq_stats, nrc_sq_disgust, by = "questionID", all = TRUE)
sq_stats <- merge(sq_stats, nrc_sq_joy, by = "questionID", all = TRUE)
sq_stats <- merge(sq_stats, nrc_sq_trust, by = "questionID", all = TRUE)
sq_stats <- merge(sq_stats, afinn_sq_avg, by = "questionID", all = TRUE)
sq_stats <- merge(sq_stats, abs_sq_count, by = "questionID", all = TRUE)
sq_stats[is.na(sq_stats)] <- 0
sq_stats <- sq_stats %>%
  mutate(anger_coef = anger_cnt /ttl_words,
         fear_coef = fear_cnt / ttl_words,
         sad_coef = sad_cnt / ttl_words,
         disgust_coef = disgust_cnt / ttl_words,
         joy_coef = joy_cnt / ttl_words,
         trust_coef = trust_cnt / ttl_words,
         abs_coef = abs_cnt / ttl_words)
sq_stats <- sq_stats %>%
  select(-c(ttl_words,anger_cnt,fear_cnt,sad_cnt,disgust_cnt,joy_cnt,trust_cnt,abs_cnt))

tmp_stats <- sq_stats[-1]

# Scale the data and perform enhanced clustering analysis again
sq_sstats <- scale(tmp_stats)
sq_dist <- get_dist(sq_sstats, method = "pearson")
head(round(as.matrix(sq_dist),3))[,1:6]
fviz_dist(sq_dist)
sq_clust <- eclust(sq_sstats,"kmeans",iter.max = 30, nstart = 25)

fviz_gap_stat(sq_clust$gap_stat)
sq_clust

msq_clust <- eclust(sq_sstats,"kmeans",iter.max = 30, k = 4, nstart = 25)
msq_clust

# ***** Random Forest classification approach *****

# Prepare the dataframe
rf_stats <- sq_stats
rf_stats$topic <- questions$topic[match(rf_stats$questionID,questions$questionID)]
rf_stats <- rf_stats[-1]
rf_stats
rf_stats$topic = factor(rf_stats$topic)

# Split data into train and test
split <- sample.split(rf_stats, SplitRatio = 0.7)
split
train <- subset(rf_stats, split == "TRUE")
test <- subset(rf_stats, split == "FALSE")

# Build the random forest model
set.seed(123)
rf_model = randomForest(x = train[-9], y = train$topic, ntree = 500)
rf_model

# Apply the model to the test dataset, show the confusion matrix, importance of
# variables, plot the model
y_pr = predict(rf_model, newdata = test[-9])
conf_matr = table(test[,9], y_pr)
conf_matr
importance(rf_model)
plot(rf_model)

# Calculate and plot average absolutist index by topic
abs_t_count <- tidy_q %>%
  filter(word %in% abs_words) %>%
  group_by(topic) %>%
  summarise(abs_cnt = n())
ttl_t_count <- tidy_q %>%
  group_by(topic) %>%
  summarise(ttl_words = n())

ttl_t_count <- merge(ttl_t_count, abs_t_count, by = "topic", all = TRUE)

ttl_t_count[is.na(ttl_t_count)] <- 0

ttl_t_count <- ttl_t_count %>%
  mutate(abs_index = abs_cnt / ttl_words)

ggplot(ttl_t_count, aes(y = fct_reorder(topic,abs_index), x = abs_index)) +
  geom_col(show.legend = FALSE, fill = 'navy') +
  labs(title = "Average Absolutist Index by Topic", x = "Absolutist Index", y = "Topics")

# ***** Topic Modeling approach *****
# References: Silge and Robinson (2016); Bansal (2024)

num_words <- tidy_sq %>%
  count(questionID,word,sort = TRUE)
num_words

# Document-term matrix
q_dtm <- num_words %>%
  cast_dtm(questionID,word,n)
q_dtm

# LDA topic model with 4 topics
q_model <- LDA(q_dtm, k=4, control = list(seed = 123))
q_model

# Find most frequent words within each topic
q_topics <- tidy(q_model, matrix = "beta")
q_topics
top_words <- q_topics %>%
  group_by(topic) %>%
  slice_max(beta,n = 10) %>%
  ungroup() %>%
  arrange(topic, -beta)
top_words
top_words %>%
  mutate(term = reorder_within(term,beta,topic)) %>%
  ggplot(aes(beta, term, fill=factor(topic))) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ topic, scales = "free") +
  scale_y_reordered()

# Per-document-per-topic probabilities (4 topics)
q_topics_g <- tidy(q_model, matrix = "gamma")
q_topics_g
q_topics_g$topicName <- questions$topic[match(q_topics_g$document,questions$questionID)]
q_topics_g
q_topics_g %>%
  mutate(topicName = reorder(topicName, gamma*topic)) %>%
  ggplot(aes(factor(topic),gamma)) +
  geom_boxplot() + 
  facet_wrap(~ topicName) + 
  labs(x = "topic", y = "gamma")

# Match calculated topics and actual topics
q_class <- q_topics_g %>%
  group_by(topicName,document) %>%
  slice_max(gamma) %>%
  ungroup()
q_class
calcName <- c("depression", "anxiety", "intimacy", "relationships")
topic <- c(1, 2, 3, 4)
q_calc <- data.frame(topic,calcName)
q_class %>%
  inner_join(q_calc, by = "topic") %>%
  filter(topicName != calcName)
q_calc_class <- q_class %>%
  inner_join(q_calc, by = "topic")
q_calc_class

# Visualize the confusion matrix
q_calc_class %>%
  count(topicName, calcName) %>%
  group_by(topicName) %>%
  mutate(pct = n / sum(n)) %>%
  ggplot(aes(calcName,topicName,fill = pct)) +
  geom_tile() +
  geom_text(aes(label = n), color = "black", fontface = 2) +
  scale_fill_gradient2(high = "darkblue", labels = percent_format()) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  labs(x = "Calculated Topics", y = "Actual Topics", fill = "Percentage")

# Perform Random Forest and Topic Modeling on two topics which resulted from 
# grouping similar topics together

# Join topics as follows: depression + anxiety, and intimacy + relationships
tidy_sq2t <- tidy_sq %>%
  mutate(topic = replace(topic, topic == "depression", "anxiety+depression"),
         topic = replace(topic, topic == "anxiety", "anxiety+depression"),
         topic = replace(topic, topic == "intimacy", "relationships"))
questions2t <- questions %>%
  mutate(topic = replace(topic, topic == "depression", "anxiety+depression"),
         topic = replace(topic, topic == "anxiety", "anxiety+depression"),
         topic = replace(topic, topic == "intimacy", "relationships"))

# Prepare the data for Random Forest
afinn_sq2t_avg <- tidy_sq2t %>% 
  inner_join(get_sentiments("afinn")) %>% 
  group_by(questionID) %>% 
  summarise(sentiment_score = mean(value))
nrc_sq2t_anger <- tidy_sq2t %>%
  inner_join(get_sentiments("nrc")) %>%
  filter(sentiment == "anger") %>%
  group_by(questionID) %>%
  summarise(anger_cnt = n())
nrc_sq2t_fear <- tidy_sq2t %>%
  inner_join(get_sentiments("nrc")) %>%
  filter(sentiment == "fear") %>%
  group_by(questionID) %>%
  summarise(fear_cnt = n())
nrc_sq2t_sad <- tidy_sq2t %>%
  inner_join(get_sentiments("nrc")) %>%
  filter(sentiment == "sadness") %>%
  group_by(questionID) %>%
  summarise(sad_cnt = n())
nrc_sq2t_disgust <- tidy_sq2t %>%
  inner_join(get_sentiments("nrc")) %>%
  filter(sentiment == "disgust") %>%
  group_by(questionID) %>%
  summarise(disgust_cnt = n())
nrc_sq2t_joy <- tidy_sq2t %>%
  inner_join(get_sentiments("nrc")) %>%
  filter(sentiment == "joy") %>%
  group_by(questionID) %>%
  summarise(joy_cnt = n())
nrc_sq2t_trust <- tidy_sq2t %>%
  inner_join(get_sentiments("nrc")) %>%
  filter(sentiment == "trust") %>%
  group_by(questionID) %>%
  summarise(trust_cnt = n())
abs_sq2t_count <- tidy_sq2t %>%
  filter(word %in% abs_words) %>%
  group_by(questionID) %>%
  summarise(abs_cnt = n())
sq2t_stats <- tidy_sq2t %>%
  group_by(questionID) %>%
  summarise(ttl_words = n())
sq2t_stats <- merge(sq2t_stats, nrc_sq2t_anger, by = "questionID", all = TRUE)
sq2t_stats <- merge(sq2t_stats, nrc_sq2t_fear, by = "questionID", all = TRUE)
sq2t_stats <- merge(sq2t_stats, nrc_sq2t_sad, by = "questionID", all = TRUE)
sq2t_stats <- merge(sq2t_stats, nrc_sq2t_disgust, by = "questionID", all = TRUE)
sq2t_stats <- merge(sq2t_stats, nrc_sq2t_joy, by = "questionID", all = TRUE)
sq2t_stats <- merge(sq2t_stats, nrc_sq2t_trust, by = "questionID", all = TRUE)
sq2t_stats <- merge(sq2t_stats, afinn_sq2t_avg, by = "questionID", all = TRUE)
sq2t_stats <- merge(sq2t_stats, abs_sq2t_count, by = "questionID", all = TRUE)
sq2t_stats[is.na(sq2t_stats)] <- 0
sq2t_stats <- sq2t_stats %>%
  mutate(anger_coef = anger_cnt /ttl_words,
         fear_coef = fear_cnt / ttl_words,
         sad_coef = sad_cnt / ttl_words,
         disgust_coef = disgust_cnt / ttl_words,
         joy_coef = joy_cnt / ttl_words,
         trust_coef = trust_cnt / ttl_words,
         abs_coef = abs_cnt / ttl_words)
sq2t_stats <- sq2t_stats %>%
  select(-c(ttl_words,anger_cnt,fear_cnt,sad_cnt,disgust_cnt,joy_cnt,trust_cnt,abs_cnt))

# Random Forest
rf2t_stats <- sq2t_stats
rf2t_stats$topic <- questions2t$topic[match(rf2t_stats$questionID,questions2t$questionID)]
rf2t_stats <- rf2t_stats[-1]
rf2t_stats
rf2t_stats$topic = factor(rf2t_stats$topic)
split2t <- sample.split(rf2t_stats, SplitRatio = 0.7)
split2t
train2t <- subset(rf2t_stats, split2t == "TRUE")
test2t <- subset(rf2t_stats, split2t == "FALSE")

# Build the model
set.seed(123)
rf2t_model = randomForest(x = train2t[-9], y = train2t$topic, ntree = 500)
rf2t_model

# Apply the model
y_pr2t = predict(rf2t_model, newdata = test2t[-9])

# Confusion matrix
conf_matr2t = table(test2t[,9], y_pr2t)
conf_matr2t
df_cm = data.frame(conf_matr2t)
df_cm %>%
  mutate(pct = Freq / sum(Freq)) %>%
  ggplot(aes(x = Var1, y = y_pr2t, fill = pct)) +
  scale_fill_gradient2(high = "darkblue", labels = percent_format()) +
  geom_tile() +
  geom_text(aes(label = Freq), color = "black", fontface = 2) +
  labs(x = "Calculated Topics", y = "Actual Topics", fill = "Percentage")

# Importance plot
importance(rf2t_model)
varImpPlot(rf2t_model, sort = TRUE, main = "Importance Plot")

# Plot the model
plot(rf2t_model)

# Drop abs_coef and see if accuracy improves
rf2t_stats2 <- rf2t_stats %>%
  select(-c(abs_coef))
split2t <- sample.split(rf2t_stats2, SplitRatio = 0.7)
split2t
train2t <- subset(rf2t_stats2, split2t == "TRUE")
test2t <- subset(rf2t_stats2, split2t == "FALSE")

set.seed(123)
rf2t_model2 = randomForest(x = train2t[-8], y = train2t$topic, ntree = 500)
rf2t_model2
y_pr2t = predict(rf2t_model2, newdata = test2t[-8])
conf_matr2t2 = table(test2t[,8], y_pr2t)
conf_matr2t2

# Topic Modeling for 2 topics
# Same steps as for topic modeling on 4 topics
num_words2t <- tidy_sq2t %>%
  count(questionID,word,sort = TRUE)
q_dtm2t <- num_words2t %>%
  cast_dtm(questionID,word,n)
q_dtm2t
q2t_model <- LDA(q_dtm2t, k=2, control = list(seed = 123))
q2t_model

q2t_topics <- tidy(q2t_model, matrix = "beta")
q2t_topics
top2t_words <- q2t_topics %>%
  group_by(topic) %>%
  slice_max(beta,n = 10) %>%
  ungroup() %>%
  arrange(topic, -beta)
top2t_words
top2t_words %>%
  mutate(term = reorder_within(term,beta,topic)) %>%
  ggplot(aes(beta, term, fill=factor(topic))) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ topic, scales = "free") +
  scale_y_reordered()
q2t_topics_g <- tidy(q2t_model, matrix = "gamma")
q2t_topics_g

q2t_topics_g$topicName <- questions2t$topic[match(q2t_topics_g$document,questions2t$questionID)]
q2t_topics_g

q2t_topics_g %>%
  mutate(topicName = reorder(topicName, gamma*topic)) %>%
  ggplot(aes(factor(topic),gamma)) +
  geom_boxplot() + 
  facet_wrap(~ topicName) + 
  labs(x = "topic", y = "gamma")
q2t_class <- q2t_topics_g %>%
  group_by(topicName,document) %>%
  slice_max(gamma) %>%
  ungroup()
q2t_class
calcName2t <- c("anxiety+depression", "relationships")
topic <- c(1, 2)
q_calc2t <- data.frame(topic,calcName2t)
q2t_class %>%
  inner_join(q_calc2t, by = "topic") %>%
  filter(topicName != calcName2t)

q2t_calc_class <- q2t_class %>%
  inner_join(q_calc2t, by = "topic")
q2t_calc_class

q2t_calc_class %>%
  count(topicName, calcName2t) %>%
  group_by(topicName) %>%
  mutate(pct = n / sum(n)) %>%
  ggplot(aes(calcName2t,topicName,fill = pct)) +
  geom_tile() +
  geom_text(aes(label = n), color = "black", fontface = 2) +
  scale_fill_gradient2(high = "darkblue", labels = percent_format()) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  labs(x = "Calculated Topics", y = "Actual Topics", fill = "Percentage")

# ***** Analyze the answers provided by psychologists *****

answers <- data.frame(chat['questionID'],chat['answerText'],chat['topic'])
summary(answers)
tidy_a <- answers %>%
  unnest_tokens(word, answerText) %>%
  anti_join(stop_words) %>%
  filter(str_detect(word,"[a-z]"))
tidy_a %>%
  filter(topic == "depression") %>%
  count(word) %>%
  with(wordcloud(word, n, max.words = 100,
                 random.order = F, random.color = F,
                 colors = c("navy", "royalblue", "turquoise")))

tidy_a %>%
  filter(topic == "anxiety") %>%
  count(word) %>%
  with(wordcloud(word, n, max.words = 100,
                 random.order = F, random.color = F,
                 colors = c("navy", "royalblue", "turquoise")))

tidy_a %>%
  filter(topic == "intimacy") %>%
  count(word) %>%
  with(wordcloud(word, n, max.words = 100,
                 random.order = F, random.color = F,
                 colors = c("navy", "royalblue", "turquoise")))

tidy_a %>%
  filter(topic == "relationships") %>%
  count(word) %>%
  with(wordcloud(word, n, max.words = 100,
                 random.order = F, random.color = F,
                 colors = c("navy", "royalblue", "turquoise")))
