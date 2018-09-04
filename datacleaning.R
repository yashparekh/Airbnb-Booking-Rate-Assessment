#read datasets
abnb_train_x<-read.csv('airbnb_train_x.csv')
abnb_test_x<-read.csv('airbnb_test_x.csv')
abnb_train_x$isTraining<-1;
abnb_test_x$isTraining<-0;
train.y<-read.csv('airbnb_train_y.csv')
abnb_train_y<-train.y[(train.y$X!=16246) & (train.y$X!=30584) &(train.y$X!=47615) &(train.y$X!=56281) &(train.y$X!=72540) &(train.y$X!=75208) &(train.y$X!=92585) &(train.y$X!=96068) &(train.y$X!=65792),]
#combine training and test data for data cleaning
abnb_full<-rbind(abnb_train_x,abnb_test_x)
abnb_full<-abnb_full[abnb_full$accommodates!='t',]
attach(abnb_full)
#clean accommodates
table(abnb_full$accommodates)
abnb_full$accommodates<-as.numeric(as.matrix(abnb_full$accommodates))
class(abnb_full$accommodates)
#clean availability variables
class(abnb_full$availability_30)
abnb_full$availability_30<-as.numeric(as.matrix(abnb_full$availability_30))
table(abnb_full$availability_30)
abnb_full$availability_60<-as.numeric(as.matrix(abnb_full$availability_60))
table(abnb_full$availability_60)
abnb_full$availability_90<-as.numeric(as.matrix(abnb_full$availability_90))
table(abnb_full$availability_90)
abnb_full$availability_365<-as.numeric(as.matrix(abnb_full$availability_365))
table(abnb_full$availability_365)
class(abnb_full$availability_365)
class(abnb_full$availability_60)
class(abnb_full$availability_90)

#clean bathrooms
table(abnb_full$bathrooms)
class(abnb_full$bathrooms)
abnb_full$bathrooms<-as.matrix(abnb_full$bathrooms)
for(i in 1:nrow(abnb_full)) {
  if(abnb_full[i,'bathrooms']==''){
    abnb_full[i,'bathrooms']<-'1'
  }
}
abnb_full$bathrooms<-as.numeric(abnb_full$bathrooms)
table(abnb_full$bathrooms)

#clean bed_type
abnb_full$bed_type<-as.factor(as.matrix(abnb_full$bed_type))
table(abnb_full$bed_type)
class(abnb_full$bed_type)

#clean bedrooms
table(abnb_full$bedrooms)
abnb_full$bedrooms<-as.matrix(abnb_full$bedrooms)
for(i in 1:nrow(abnb_full)) {
  if(abnb_full[i,'bedrooms']==''){
    abnb_full[i,'bedrooms']<-'1'
  }
} #replace null values with majority class
abnb_full$bedrooms<-as.numeric(abnb_full$bedrooms)
table(abnb_full$bedrooms)

#clean beds
table(abnb_full$beds)
abnb_full$beds<-as.factor(abnb_full$beds)
abnb_full$beds<-as.numeric(as.matrix(abnb_full$beds))
table(abnb_full$beds)
class(abnb_full$beds)

#clean cancellation_policy
table(abnb_full$cancellation_policy)
abnb_full$cancellation_policy<-as.matrix(abnb_full$cancellation_policy)
table(abnb_full$cancellation_policy)
abnb_full$cancellation_policy<-as.factor(abnb_full$cancellation_policy)

#clean city_name
table(abnb_full$city_name)
abnb_full$city_name<-as.matrix(abnb_full$city_name)
abnb_full$city_name<-as.factor(abnb_full$city_name)

#clean guests_included
table(abnb_full$guests_included)
class(abnb_full$guests_included)

#clean host variables
table(abnb_full$host_has_profile_pic)
abnb_full$host_has_profile_pic<-as.matrix(abnb_full$host_has_profile_pic)
for(i in 1:nrow(abnb_full)) {
  if(abnb_full[i,'host_has_profile_pic']==''){
    abnb_full[i,'host_has_profile_pic']<-'t'
  }
} #replace null values with majority class
abnb_full$host_has_profile_pic<-as.factor(abnb_full$host_has_profile_pic)
table(abnb_full$host_has_profile_pic)

#####
table(abnb_full$host_identity_verified)
abnb_full$host_identity_verified<-as.matrix(abnb_full$host_identity_verified)
for(i in 1:nrow(abnb_full)) {
  if(abnb_full[i,'host_identity_verified']==''){
    abnb_full[i,'host_identity_verified']<-'t'
  }
} #replace null values with majority class
table(abnb_full$host_identity_verified)
abnb_full$host_identity_verified<-as.factor(abnb_full$host_identity_verified)

#####
table(abnb_full$host_listings_count)
abnb_full$host_listings_count<-as.matrix(abnb_full$host_listings_count)
for(i in 1:nrow(abnb_full)) {
  if(abnb_full[i,'host_listings_count']==''){
    abnb_full[i,'host_listings_count']<-'1'
  }
} #replace null values with majority class
abnb_full$host_listings_count<-as.numeric(abnb_full$host_listings_count)
table(abnb_full$host_listings_count)

#####clean host_is_superhost
table(abnb_full$host_is_superhost)
abnb_full$host_is_superhost<-as.matrix(abnb_full$host_is_superhost)
for(i in 1:nrow(abnb_full)) {
  if(abnb_full[i,'host_is_superhost']==''){
    abnb_full[i,'host_is_superhost']<-'f'
  }
} 
abnb_full$host_is_superhost<-as.factor(abnb_full$host_is_superhost)
table(abnb_full$host_is_superhost)

#clean instant_bookable
table(abnb_full$instant_bookable)
abnb_full$instant_bookable<-as.matrix(abnb_full$instant_bookable)
for(i in 1:nrow(abnb_full)) {
  if(abnb_full[i,'instant_bookable']==''){
    abnb_full[i,'instant_bookable']<-'f'
  }
} #replace null values with majority class
abnb_full$instant_bookable<-as.factor(abnb_full$instant_bookable)
table(abnb_full$instant_bookable)

#clean is_location_exact
table(abnb_full$is_location_exact)
abnb_full$is_location_exact<-as.matrix(abnb_full$is_location_exact)
for(i in 1:nrow(abnb_full)) {
  if(abnb_full[i,'is_location_exact']==''){
    abnb_full[i,'is_location_exact']<-'t'
  }
} #replace null values with majority class
abnb_full$is_location_exact<-as.factor(abnb_full$is_location_exact)
table(abnb_full$is_location_exact)

#clean license
table(abnb_full$license)
abnb_full$license<-as.matrix(abnb_full$license)
for(i in 1:nrow(abnb_full)) {
  if(abnb_full[i,'license']==''){
    abnb_full[i,'license']<-'f'
  }
  else {
    abnb_full[i,'license']<-'t'
  }
} 
table(abnb_full$license)
abnb_full$license<-as.factor(abnb_full$license)
#clean min nights
table(abnb_full$minimum_nights)
class(abnb_full$minimum_nights)

#clean price 
table(abnb_full$price)
class(abnb_full$price)

#clean property_type
table(abnb_full$property_type)
abnb_full$property_type<-as.matrix(abnb_full$property_type)
for(i in 1:nrow(abnb_full)) {
  if(abnb_full[i,'property_type']==''){
    abnb_full[i,'property_type']<-'House'
  }
} #replace null values with majority class
table(abnb_full$property_type)
abnb_full$property_type<-as.factor(abnb_full$property_type)

#clean room_type
table(abnb_full$room_type)
abnb_full$room_type<-as.matrix(abnb_full$room_type)
for(i in 1:nrow(abnb_full)) {
  if(abnb_full[i,'room_type']==''){
    abnb_full[i,'room_type']<-'Entire home/apt'
  }
} #replace null values with majority class
table(abnb_full$room_type)
abnb_full$room_type<-as.factor(abnb_full$room_type)

train.x<-abnb_full[(abnb_full$isTraining==1),]
test.x<-abnb_full[(abnb_full$isTraining==0),]
train.y<-abnb_train_y[,2:3]

final.train.x<-train.x[,2:71]
final.test.x<-test.x[,2:71]

write.csv(final.train.x, file = "clean_training.csv")
write.csv(final.test.x, file = "clean_test.csv")
write.csv(train.y, file = 'clean_train_y.csv')

