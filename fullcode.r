
################################
# Create edx set, validation set
################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data

set.seed(1, sample.kind="Rounding")
# if using R 3.5 or earlier, use `set.seed(1)` instead
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]  # train set
temp <- movielens[test_index,]  # test set, which needs to be updated to make sure userid and movieid in validation

# Make sure userId and movieId in validation set are also in edx set

validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set

removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)


#  Creating a movie recommendation system using the MovieLens dataset

#  Train a machine learning algorithm using the inputs in one subset 
#  to predict movie ratings in the validation set. Your project itself
#  will be assessed by peer grading


# 1 - Find average for all movies in training set
# 2. Incorporate the Movie effect
# 3. Incorporate the User effect

edx %>% 
  dplyr::count(movieId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "black") + 
  scale_x_log10() + 
  ggtitle("Movies")

test_index <- createDataPartition(y = edx$rating, times = 1, p = .1, list = FALSE)
edx_train <- edx[-test_index,]  # train set
temp <- edx[test_index,]  # test set, which needs to be updated to make sure userid and movieid in validation

edx_test <- temp %>% 
  semi_join(edx_train, by = "movieId") %>%
  semi_join(edx_train, by = "userId")



# Root Mean Square Error function. used to evaluate the accuracy of each algorithm's predictions
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

# average of all movies 
mu_hat <- mean(edx_train$rating)
naive_rmse <- RMSE(edx_test$rating, mu_hat)
rmse_results <- data_frame(method = "Just Average", RMSE = naive_rmse)

# Prove average of all movies by picking arbitrary value 2.5 to be estimate
test_predictions <- rep(2.5, nrow(edx_test))
arbitrary_rmse <- RMSE(edx_test$rating, test_predictions)

rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Arbitrary Value",
                                     RMSE = arbitrary_rmse ))
table1 <- rmse_results
rmse_results %>% knitr::kable()

# 2. Incorporate the Movie effect into the algorithm, predict ratings, then evaluate the RMSE
movie_average <- edx_train %>% group_by(movieId) %>% summarize(b_i = mean(rating - mu_hat))
# view disparity in average movie ratings per movie
movie_average %>% qplot(b_i, geom ="histogram", bins = 10, data = ., color = I("black"), ylab = "count")

predicted_ratings <- mu_hat + edx_test %>% 
  left_join(movie_average, by='movieId') %>%
  .$b_i

# RMSE(true rating, predicted rating)
movie_rmse_results <- RMSE(edx_test$rating, predicted_ratings)

rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Movie Effect",
                                     RMSE = movie_rmse_results ))

rmse_results %>% knitr::kable()


# 3. Incorporate the User Effect into the algorithm, predict ratings, then evaluate the RMSE
# view disparity in average user ratings per movie
edx_train %>% group_by(userId) %>% summarize(b_u = mean(rating)) %>% filter(n() >= 50) %>%
  ggplot(aes(b_u)) + geom_histogram(bins = 20, color = "black")

user_average <- edx_train %>% 
  left_join(movie_average, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu_hat - b_i))

predicted_ratings <- edx_test %>% 
  left_join(movie_average, by='movieId') %>%
  left_join(user_average, by='userId') %>%
  mutate(prediction = mu_hat + b_i + b_u) %>%
  .$prediction

rmse <- RMSE(edx_test$rating, predicted_ratings )

rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Movie + User Effect",
                                     RMSE = rmse ))
rmse_results %>% knitr::kable()

# view predictions based on movie and user effect.
final_pred <- edx_test %>% 
  left_join(movie_average, by='movieId') %>%
  left_join(user_average, by='userId') %>%
  mutate(prediction = mu_hat + b_i + b_u)

head(final_pred %>% select(title, prediction))



# 4. Incorporate the Genre Effect into the algorithm, predict ratings, then evaluate the RMSE
# different genres have a different average rating
edx_train %>% group_by(genres) %>% summarize(b_g = mean(rating)) %>% filter(n() >= 50) %>%
  ggplot(aes(b_g)) + geom_histogram(bins = 20, color = "black")

genre_average<- edx_train %>% 
  left_join(movie_average, by='movieId') %>%
  left_join(user_average, by = 'userId') %>%
  group_by(genres) %>%
  summarize(b_g = mean(rating - mu_hat - b_i - b_u))

predicted_ratings <- edx_test %>% 
  left_join(movie_average, by='movieId') %>%
  left_join(user_average, by='userId') %>%
  left_join(genre_average, by='genres') %>%
  mutate(prediction = mu_hat + b_i + b_u + b_g) %>%
  .$prediction

rmse <- RMSE(edx_test$rating, predicted_ratings)

rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Movie + User + Genre Effect",
                                     RMSE = rmse ))
rmse_results %>% knitr::kable()


# since some of the genres have a very low movie count, we will need to account for the high variation associated 
# with low movie accounts by regularizing lambda
# by incorporating lambda
# do a regularized version for Movie + User + Genre
# this uses lambda. spoken about here 
# https://rafalab.github.io/dsbook/large-datasets.html#recommendation-systems
# we'll minimize lambda for the model including movie, user, and genre effects

edx_train %>%
  group_by(movieId,title) %>%
  summarize(avg_rating = mean(rating), num = n()) %>%
  arrange(desc(avg_rating))%>% select(title, avg_rating, num)  %>% top_n(avg_rating,5) 

# set lambda to an arbitrary value. 3 was picked
lambda <- 3

reg_movie_average <- edx_train %>% group_by(movieId) %>% summarize(b_i = sum(rating - mu_hat)/(n()+lambda), n_i = n()) 

# The  regularized version dampens the weighting for movies with fewer ratings
tibble(original = movie_average$b_i, 
       regularlized = reg_movie_average$b_i, 
       n = reg_movie_average$n_i) %>%
  ggplot(aes(original, regularlized, size=sqrt(n))) + 
  geom_point(shape=1, alpha=0.5)

reg_user_avg <- edx_train %>% 
  left_join(reg_movie_average, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - mu_hat - b_i)/(n()+lambda), n_i = n())

reg_genre_avg <- edx_train %>% 
  left_join(reg_movie_average, by='movieId') %>%
  left_join(reg_user_avg, by = 'userId') %>%
  group_by(genres) %>%
  summarize(b_g = sum(rating - mu_hat - b_i - b_u)/(n()+lambda), n_i = n())

reg_predicted_ratings <- edx_test %>% 
  left_join(reg_movie_average, by='movieId') %>%
  left_join(reg_user_avg, by='userId') %>%
  left_join(reg_genre_avg, by='genres') %>%
  mutate(prediction = mu_hat + b_i + b_u + b_g) %>%
  .$prediction

rmse <- RMSE(edx_test$rating, reg_predicted_ratings)

rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Regularized (Lambda = 3) Movie + User + Genre Effect",
                                     RMSE = rmse ))
rmse_results %>% knitr::kable()

#  considering Lambda is a tuning parameter, we can use cross-validation to choose the best value of lambda
lambdas <- seq(4, 5, 0.1)
# build a function to calculate the RMSE for each value of lambda in the lambda vector
# the lowest RMSE is the optimized Lambda
rmses <- sapply(lambdas, function(lambda){
  mv_avg <- edx_train %>% 
    group_by(movieId) %>% 
    summarize(b_i = sum(rating - mu_hat)/(n()+lambda), n_i = n()) 
  
  u_avg <- edx_train %>% 
    left_join(mv_avg, by='movieId') %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - mu_hat - b_i)/(n()+lambda), n_i = n())
  
  g_avg <- edx_train %>% 
    left_join(mv_avg, by='movieId') %>%
    left_join(u_avg, by = 'userId') %>%
    group_by(genres) %>%
    summarize(b_g = sum(rating - mu_hat - b_i - b_u)/(n()+lambda), n_i = n())
  
  preds <- edx_test %>% 
    left_join(mv_avg, by='movieId') %>%
    left_join(u_avg, by='userId') %>%
    left_join(g_avg, by='genres') %>%
    mutate(prediction = mu_hat + b_i + b_u + b_g) %>%
    .$prediction
  return(RMSE(edx_test$rating, preds))
})
qplot(lambdas, rmses)  
lambdas[which.min(rmses)]


lambda <- lambdas[which.min(rmses)]

reg_movie_average2 <- edx_train %>% group_by(movieId) %>% summarize(b_i = sum(rating - mu_hat)/(n()+lambda), n_i = n()) 

reg_user_avg2 <- edx_train %>% 
  left_join(reg_movie_average2, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - mu_hat - b_i)/(n()+lambda), n_i = n())

reg_genre_avg2 <- edx_train %>% 
  left_join(reg_movie_average2, by='movieId') %>% 
  left_join(reg_user_avg2, by = 'userId') %>% 
  group_by(genres) %>%
  summarize(b_g = sum(rating - mu_hat - b_i - b_u)/(n()+lambda), n_i = n())

reg_predicted_ratings2 <- edx_test %>% 
  left_join(reg_movie_average2, by='movieId') %>%
  left_join(reg_user_avg2, by='userId') %>%
  left_join(reg_genre_avg2, by='genres') %>%
  mutate(prediction = mu_hat + b_i + b_u + b_g) %>%
  .$prediction

rmse <- RMSE(edx_test$rating, reg_predicted_ratings2)

rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Regularized (Lambda = 4.4) Movie + User + Genre Effect",
                                     RMSE = rmse ))
rmse_results %>% knitr::kable()


#"For a final test of your algorithm, predict movie ratings in the validation set"

validation_predicted_ratings <- validation %>% 
  left_join(reg_movie_average2, by='movieId') %>%
  left_join(reg_user_avg2, by='userId') %>%
  left_join(reg_genre_avg2, by='genres') %>%
  mutate(prediction = mu_hat + b_i + b_u + b_g) %>%
  .$prediction


# Could not run RMSE function as it did not account for Null ratings, and returned NA
# The null ratings were created because the edx dataset had only one record for each movie. 
# Due to this, the training and test set do not contain either of the two movies, so they were not trained.
# When using the validation set, we needed to set the mean to ignore NA values, then the RMSE could be calculated.

NA_Index <- which(is.na(validation_predicted_ratings))
na_movies <- validation[NA_Index,]$movieId
edx %>% filter(movieId %in% c(na_movies))
edx_train %>% filter(movieId %in% c(na_movies))
edx_test %>% filter(movieId %in% c(na_movies))

validation %>% filter(movieId %in% c(na_movies))


# Calculate final RMSE:
FINAL_RMSE <- sqrt(mean((validation$rating - validation_predicted_ratings)^2,na.rm = TRUE))
FINAL_RMSE

final_rmse_results <- bind_rows(rmse_results,
                                data_frame(method="VALIDATION: Regularized (Lambda = 4.4) Movie + User + Genre Effect",
                                           RMSE = FINAL_RMSE ))

final_rmse_results %>% knitr::kable()
