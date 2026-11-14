# Run from the milestone folder. All transformations are defined here.
options(stringsAsFactors = FALSE)
for (p in c('data/cleaned','data/handoff','outputs','figures')) dir.create(p, recursive=TRUE, showWarnings=FALSE)
raw <- read.csv('data/raw/framingham.csv', na.strings=c('NA','','N/A','?'))
stopifnot(nrow(raw)==4240L, ncol(raw)==16L)
raw$source_row <- seq_len(nrow(raw))
write_out <- function(x, name) write.csv(x, file.path('outputs',name), row.names=FALSE, na='')
missing_table <- data.frame(variable=names(raw), missing=colSums(is.na(raw)), percent=100*colMeans(is.na(raw)))
write_out(missing_table,'missingness_raw.csv')
binary <- c('male','currentSmoker','BPMeds','prevalentStroke','prevalentHyp','diabetes','TenYearCHD')
checks <- data.frame(check=c('Exact duplicate original rows','Invalid binary codes','Invalid education codes','Nonpositive medical measurements','Negative cigarettes/day','Systolic <= diastolic','Nonsmoker with positive cigarettes','Smoker with zero cigarettes'),
 count=c(sum(duplicated(raw[setdiff(names(raw),'source_row')])),
 sum(sapply(raw[binary],function(x)sum(!is.na(x)&!x%in%c(0,1)))),
 sum(!is.na(raw$education)&!raw$education%in%1:4),
 sum(sapply(raw[c('age','totChol','sysBP','diaBP','BMI','heartRate','glucose')],function(x)sum(x<=0,na.rm=TRUE))),
 sum(raw$cigsPerDay<0,na.rm=TRUE),sum(raw$sysBP<=raw$diaBP,na.rm=TRUE),
 sum(raw$currentSmoker==0&raw$cigsPerDay>0,na.rm=TRUE),sum(raw$currentSmoker==1&raw$cigsPerDay==0,na.rm=TRUE)))
write_out(checks,'validity_checks.csv')
# Stop on definite schema/range errors; report smoking combinations without rewriting.
stopifnot(all(checks$count[1:6]==0))
removed <- raw[is.na(raw$glucose),]
observed <- raw[!is.na(raw$glucose),]
write_out(removed,'excluded_missing_glucose.csv')
selection <- do.call(rbind,lapply(c('age','male','currentSmoker','BMI','sysBP','diabetes'),function(v){
 a<-raw[!is.na(raw$glucose),v]; b<-raw[is.na(raw$glucose),v]
 data.frame(variable=v,observed_target_mean=mean(a,na.rm=TRUE),missing_target_mean=mean(b,na.rm=TRUE),
 standardized_difference=(mean(a,na.rm=TRUE)-mean(b,na.rm=TRUE))/sqrt((var(a,na.rm=TRUE)+var(b,na.rm=TRUE))/2))}))
write_out(selection,'target_missingness_comparison.csv')
pattern <- as.data.frame(table(rowSums(is.na(raw[setdiff(names(raw),'source_row')]))))
names(pattern)<-c('missing_fields_per_row','rows');write_out(pattern,'missing_fields_per_row.csv')

# Deterministic split membership before fitting preprocessing. This is an
# exploratory holdout: prior project inspection used all rows, as disclosed.
set.seed(262)
train_rows <- sort(sample(seq_len(nrow(observed)),floor(.8*nrow(observed))))
split <- rep('test',nrow(observed));split[train_rows]<-'train'
base <- observed[setdiff(names(observed),c('diabetes','TenYearCHD'))]
train <- base[split=='train',];test <- base[split=='test',]
write.csv(train,'data/handoff/train_unimputed.csv',row.names=FALSE,na='')
write.csv(test,'data/handoff/test_unimputed.csv',row.names=FALSE,na='')
write_out(data.frame(source_row=base$source_row,split=split),'split_manifest.csv')

mode_value <- function(x) as.numeric(names(sort(table(x),decreasing=TRUE))[1])
fit_preprocessor <- function(d){
 # Numeric distances use continuous measurements plus fixed 0/1 indicators.
 # Education codes and medication unknowns are excluded from distances.
 distance_cols <- c('male','age','currentSmoker','cigsPerDay','prevalentStroke','prevalentHyp','totChol','sysBP','diaBP','BMI','heartRate')
 mu<-sapply(d[distance_cols],mean,na.rm=TRUE);sdv<-sapply(d[distance_cols],sd,na.rm=TRUE);sdv[!is.finite(sdv)|sdv==0]<-1
 list(donors=d,cols=distance_cols,mu=mu,sd=sdv,
      heart=median(d$heartRate,na.rm=TRUE),education=mode_value(d$education),
      cigarettes=median(d$cigsPerDay[d$currentSmoker==1],na.rm=TRUE),k=10L)
}
knn_values <- function(query,fit,target){
 donors<-fit$donors;cols<-setdiff(fit$cols,target)
 z<-sweep(sweep(as.matrix(donors[cols]),2,fit$mu[cols],'-'),2,fit$sd[cols],'/')
 vapply(seq_len(nrow(query)),function(i){
  q<-(as.numeric(query[i,cols])-fit$mu[cols])/fit$sd[cols]
  delta<-sweep(z,2,q,'-');available<-rowSums(!is.na(delta))
  dist<-rowSums(delta^2,na.rm=TRUE)*length(cols)/pmax(available,1)
  dist[available==0|is.na(donors[[target]])]<-Inf
  ids<-head(order(dist,donors$source_row),fit$k);ids<-ids[is.finite(dist[ids])]
  if(length(ids))mean(donors[[target]][ids]) else median(donors[[target]],na.rm=TRUE)
 },numeric(1))
}
apply_preprocessor <- function(d,fit){
 out<-d
 for(v in c('education','BMI','totChol','cigsPerDay','heartRate'))out[[paste0(v,'_missing')]]<-as.integer(is.na(d[[v]]))
 # Both KNN imputations use original queries/donors, avoiding order effects.
 for(v in c('BMI','totChol')){
  ids<-which(is.na(d[[v]]));if(length(ids))out[ids,v]<-knn_values(d[ids,],fit,v)
 }
 out$heartRate[is.na(out$heartRate)]<-fit$heart
 out$education[is.na(out$education)]<-fit$education
 ids<-which(is.na(out$cigsPerDay));out$cigsPerDay[ids]<-ifelse(out$currentSmoker[ids]==0,0,fit$cigarettes)
 out$BPMeds<-ifelse(is.na(d$BPMeds),'Unknown',as.character(d$BPMeds))
 # Candidate derived feature; omitted from the primary modeling CSV because it
 # is an exact linear combination of sysBP and diaBP.
 stopifnot(!anyNA(out),all(out$glucose==d$glucose),all(out$education%in%1:4),all(out$BPMeds%in%c('0','1','Unknown')))
 out
}
fit<-fit_preprocessor(train)
clean<-apply_preprocessor(base,fit)
saveRDS(fit,'outputs/preprocessor.rds')
write.csv(clean,'data/cleaned/framingham_cleaned.csv',row.names=FALSE)
write.csv(clean[split=='train',],'data/handoff/train_cleaned.csv',row.names=FALSE)
write.csv(clean[split=='test',],'data/handoff/test_cleaned.csv',row.names=FALSE)
candidate<-data.frame(source_row=clean$source_row,pulse_pressure=clean$sysBP-clean$diaBP)
write.csv(candidate,'data/cleaned/candidate_pulse_pressure.csv',row.names=FALSE)
write_out(data.frame(parameter=c('heartRate_median','education_mode','smoker_cigarettes_median','knn_k'),value=c(fit$heart,fit$education,fit$cigarettes,fit$k)),'imputation_parameters.csv')

# Cell-level audit: every filled value retains its original source row.
audit<-do.call(rbind,lapply(c('heartRate','education','BMI','totChol','cigsPerDay','BPMeds'),function(v){
 ids<-which(is.na(base[[v]]));data.frame(source_row=base$source_row[ids],variable=v,old_value=NA_character_,new_value=as.character(clean[ids,v]))}))
write_out(audit,'imputed_cells.csv')
numeric_vars<-c('age','cigsPerDay','totChol','sysBP','diaBP','BMI','heartRate','glucose')
summaries<-do.call(rbind,lapply(numeric_vars,function(v){x<-train[[v]];x<-x[!is.na(x)];q<-quantile(x,c(.25,.5,.75));s<-sd(x)
data.frame(variable=v,n=length(x),mean=mean(x),sd=s,min=min(x),q1=q[1],median=q[2],q3=q[3],max=max(x),skewness=mean((x-mean(x))^3)/s^3)}))
rownames(summaries)<-NULL;write_out(summaries,'training_numerical_summary.csv')
flags<-do.call(rbind,lapply(numeric_vars,function(v){x<-observed[[v]];q<-quantile(train[[v]],c(.25,.75),na.rm=TRUE);iq<-diff(q);lo<-q[1]-1.5*iq;hi<-q[2]+1.5*iq
ids<-which(x<lo|x>hi);data.frame(source_row=observed$source_row[ids],variable=rep(v,length(ids)),value=x[ids],lower=rep(as.numeric(lo),length(ids)),upper=rep(as.numeric(hi),length(ids)))}))
write_out(flags,'outlier_flags_retained.csv')
ct<-clean[split=='train',]
continuous<-c('age','cigsPerDay','totChol','sysBP','diaBP','BMI','heartRate')
correlation<-cor(ct[continuous]);write.csv(correlation,'outputs/pearson_predictor_matrix.csv')
pairs<-which(upper.tri(correlation),arr.ind=TRUE)
pair_table<-data.frame(feature1=rownames(correlation)[pairs[,1]],feature2=colnames(correlation)[pairs[,2]],r=correlation[pairs])
write_out(pair_table[order(-abs(pair_table$r)),],'pearson_pairs.csv')
vif<-data.frame(variable=continuous,VIF=sapply(continuous,function(v){m<-lm(reformulate(setdiff(continuous,v),response=v),data=ct);1/(1-summary(m)$r.squared)}))
write_out(vif,'continuous_predictor_vif.csv')
review_pairs<-data.frame(feature1=c('sysBP','currentSmoker','prevalentHyp','prevalentHyp'),feature2=c('diaBP','cigsPerDay','sysBP','diaBP'))
review_pairs$r<-mapply(function(a,b)cor(ct[[a]],ct[[b]]),review_pairs$feature1,review_pairs$feature2)
review_pairs$decision<-'Retain both provisionally'
review_pairs$reason<-c('Distinct pressure measurements; inspect VIF and validate alternatives.',
 'Smoking status and daily intensity have different interpretations; test whether status adds value.',
 'Recorded hypertension status and current pressure need not coincide; confirm definition.',
 'Recorded hypertension status and current pressure need not coincide; confirm definition.')
write_out(review_pairs,'collinearity_decisions.csv')
# Extend the diagnostic to the encoded primary predictor design, excluding ID,
# target, and constant columns. Values are per-column VIFs, not categorical GVIFs.
design_data<-ct[setdiff(names(ct),c('source_row','glucose'))]
design_data$education<-factor(design_data$education,levels=1:4)
design_data$BPMeds<-factor(design_data$BPMeds,levels=c('0','1','Unknown'))
design<-model.matrix(~.,data=design_data)[,-1,drop=FALSE]
constant_design<-colnames(design)[apply(design,2,function(x)length(unique(x))<2)]
design<-design[,apply(design,2,function(x)length(unique(x))>1),drop=FALSE]
design_vif<-data.frame(encoded_column=colnames(design),VIF=vapply(seq_len(ncol(design)),function(j){
 y<-design[,j];x<-cbind(1,design[,-j,drop=FALSE]);m<-lm.fit(x,y);r2<-1-sum(m$residuals^2)/sum((y-mean(y))^2)
 if(1-r2<1e-12)Inf else 1/(1-r2)
},numeric(1)))
write_out(design_vif,'encoded_design_vif.csv')
redundancy_check<-data.frame(check='BPMeds_missing equals BPMeds Unknown dummy',exact_duplicate=all(as.integer(is.na(train$BPMeds))==as.integer(ct$BPMeds=='Unknown')),decision='Remove BPMeds_missing; retain Unknown category')
write_out(redundancy_check,'removed_redundant_feature.csv')
association<-data.frame(variable=continuous,pearson=sapply(continuous,function(v)cor(ct[[v]],ct$glucose)),spearman=sapply(continuous,function(v)cor(ct[[v]],ct$glucose,method='spearman')))
write_out(association,'training_target_associations.csv')
education_spearman<-cor(ct$education[ct$education_missing==0],ct$glucose[ct$education_missing==0],method='spearman')
category_counts<-do.call(rbind,lapply(c('male','education','currentSmoker','BPMeds','prevalentStroke','prevalentHyp'),function(v){t<-table(ct[[v]]);data.frame(variable=v,level=names(t),count=as.integer(t),percent=100*as.numeric(t)/sum(t))}))
write_out(category_counts,'training_category_counts.csv')
categorical_overlap<-function(a,b){
 tab<-table(a,b);p<-tab/sum(tab);pa<-rowSums(p);pb<-colSums(p)
 expected<-outer(pa,pb);nz<-p>0;mi<-sum(p[nz]*log(p[nz]/expected[nz]));ha<- -sum(pa[pa>0]*log(pa[pa>0]));hb<- -sum(pb[pb>0]*log(pb[pb>0]))
 test<-suppressWarnings(chisq.test(tab));data.frame(chi_square=unname(test$statistic),p_value=test$p.value,NMI=if(ha+hb>0)2*mi/(ha+hb) else 0,min_expected=min(test$expected))
}
overlap<-categorical_overlap(ct$prevalentHyp,ct$BPMeds);write_out(overlap,'categorical_overlap.csv')
png_plot<-function(name,code,width=1100,height=750){png(file.path('figures',name),width=width,height=height,res=120);on.exit(dev.off());force(code)}
png_plot('missingness.png',{m<-missing_table[missing_table$missing>0,];par(mar=c(8,4,2,1));barplot(m$percent,names.arg=m$variable,las=2,ylab='Missing (%)',main='Raw dataset missingness')})
png_plot('glucose_distribution.png',{par(mfrow=c(1,2));hist(train$glucose,breaks=35,main='Training glucose distribution',xlab='Glucose (reported units: mg/dL)');boxplot(train$glucose,main='Training glucose',ylab='Glucose')})
png_plot('predictor_distributions.png',{par(mfrow=c(2,4),mar=c(4,4,3,1));for(v in numeric_vars)hist(train[[v]],main=v,xlab=v,col='grey85')},1400,800)
png_plot('glucose_relationships.png',{par(mfrow=c(2,3));for(v in c('age','BMI','sysBP','diaBP','totChol','cigsPerDay'))plot(ct[[v]],ct$glucose,pch=16,cex=.35,col=adjustcolor('black',.25),xlab=v,ylab='glucose')},1300,850)
png_plot('correlations.png',{par(mar=c(8,8,3,2));image(seq_along(continuous),seq_along(continuous),correlation,zlim=c(-1,1),col=colorRampPalette(c('steelblue','white','firebrick'))(101),axes=FALSE,xlab='',ylab='',main='Training predictor Pearson correlations');axis(1,seq_along(continuous),continuous,las=2);axis(2,seq_along(continuous),continuous,las=2);for(i in seq_along(continuous))for(j in seq_along(continuous))text(i,j,round(correlation[i,j],2),cex=.7)})
png_plot('categorical_glucose.png',{par(mfrow=c(1,3));for(v in c('currentSmoker','education','BPMeds'))boxplot(ct$glucose~factor(ct[[v]]),main=v,xlab=v,ylab='glucose')},1300,600)
png_plot('blood_pressure_overlap.png',{plot(ct$sysBP,ct$diaBP,pch=16,cex=.5,col=adjustcolor('black',.25),xlab='Systolic BP',ylab='Diastolic BP',main='Related, distinct blood-pressure measurements')})
flow<-data.frame(stage=c('Raw rows','Excluded missing glucose','Retained observed targets','Training rows','Test rows','Imputed cells','Final columns'),n=c(nrow(raw),nrow(removed),nrow(clean),sum(split=='train'),sum(split=='test'),nrow(audit),ncol(clean)))
write_out(flow,'cleaning_flow.csv')
capture.output(sessionInfo(),file='outputs/sessionInfo.txt')
cat('Completed:',nrow(clean),'rows;',ncol(clean),'columns;',nrow(audit),'filled cells.\n')
