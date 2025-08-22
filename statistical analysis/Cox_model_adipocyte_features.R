#install.packages(c("survival", "survminer"))

library(tidyverse)
library(survival)
library(survminer)
library(openxlsx)
library(forestplot)
# ========= FUNCTIONS ========== 

save_plot <- function(filename){
  
  ggsave(
    filename,
    plot = last_plot(),
    device = NULL,
    path = NULL,
    scale = 1,
    width = NA,
    height = NA,
    units = c("in", "cm", "mm", "px"),
    dpi = 300,
    limitsize = TRUE,
    bg = 'white'
  )
}

multiple_univariate_cox <- function(data, time, event, covariates, img_path){
  
  #dir.create(img_path, recursive = TRUE)
  results <- data.frame()
  for(i in 1:length(covariates)){
    
    data_complete <- cbind(data[time], data[event], data[covariates[i]])
    data_complete <- data_complete[complete.cases(data_complete), ]
    # ------ z-score data ------
    data_complete <- data_complete %>%
      mutate_at(vars(-matches(paste0(time, "|", event))), scale)
    # *******************************
    colnames(data_complete)[3] <- covariates[i]
    formula <- as.formula(paste('Surv(', time,  ', ', event, ') ~ ', covariates[i], sep = ''))
    print(covariates[i])
    res.cox <- coxph(formula, data = data_complete)
    z1 <- cox.zph(res.cox)
    schoenfeld_test_p <- z1$table[1,3]
    sum <- summary(res.cox)
    summary_results <- data.frame(sum$coefficients, sum$conf.int, sum$waldtest[3], sum$logtest[3], sum$sctest[3], schoenfeld_test_p)
    results <- rbind(results, summary_results)
    #if(summary_results$Pr...z.. < 0.05){
    ggsurvplot(
      survfit(res.cox, data = data_complete),
      data = data_complete,
      palette = "#2E9FDF",
      title = covariates[i],
      ggtheme = theme_minimal() +
        theme(plot.title = element_text(hjust = 0.5)) # Centering the title
    )
    save_plot(paste0(img_path, "/COX_", covariates[i], ".png"))
    #}
  }
  colnames(results)[10:12] <- c('waldtest_pvalue', 'likelihoodtest_pvalue', 'sctest(logrank)test_pvalue')
  colnames(results)[1:2] <- c('beta coef', 'HR')
  colnames(results)[5] <- c('p_value')
  return(results)
  
}

multivariate_cox <- function(data, time, event, covariates){
  
  data_complete <- cbind(data[time], data[event], data[covariates])
  data_complete <- data_complete[complete.cases(data_complete), ]
  # ------ z-score data ------
  data_complete <- data_complete %>%
    mutate_at(vars(-matches(paste0(time, "|", event))), scale)
  # *******************************
  
  covariates_combined <- paste(covariates, collapse = '+')
  formula <- as.formula(paste('Surv(', time,  ', ', event, ') ~ ', covariates_combined, sep = ''))
  res.cox <- coxph(formula, data =  data_complete)
  
  
  # forest_plot(res.cox,
  #             factor_labeller = covariate_names,
  #             endpoint_labeller = c(time="OS"),
  #             orderer = ~order(HR),
  #             labels_displayed = c("endpoint", "factor", "n"),
  #             ggtheme = ggplot2::theme_bw(base_size = 10),
  #             relative_widths = c(1, 1.5, 1),
  #             HR_x_breaks = c(0.25, 0.5, 0.75, 1, 1.5, 2))
  
  sum <- summary(res.cox)
  
  summary_results <- data.frame(sum$coefficients, sum$conf.int, sum$waldtest[3], sum$logtest[3], sum$sctest[3])
  colnames(summary_results)[10:12] <- c('waldtest_pvalue', 'likelihoodtest_pvalue', 'sctest(logrank)test_pvalue')
  colnames(summary_results)[1:2] <- c('beta coef', 'HR')
  colnames(summary_results)[5] <- c('p_value')
  return(res.cox)
  
}

# ========= LOAD DATA ========== 
AllData <- read.csv(paste("C:/_research_projects/Adipocyte model project/Adipocyte analysis/PLCO/","PLCO_clin_FATONLY_CaFtSt_WSI_features_aggregated.csv", sep = ''), stringsAsFactors = TRUE, na.strings=c("nd", "ND", "U", "", "na", "NaN"))%>%
  print()

#OM3
covariates <- colnames(AllData[,40:length(AllData)])
time <- 'Overall.Survival..Time.to.Death.or.to.last.survival.status.if_1'
event <- 'Patient.status..1.dead..0.alive.'
pth = 'C:/_research_projects/Adipocyte model project/Adipocyte analysis/Omental Mets/DeepLabV3plus/DeepLabV3plus fat tissue/Cox analysis'
dir.create(pth, recursive = TRUE)

res <- multiple_univariate_cox(AllData, time, event, covariates, paste0(pth, '/OM3 DLV-fat_tissue_Cox plots'))
write.xlsx(res, file = paste0(pth,'/DLV-fat_OM3_univariate_cox.xlsx'), rowNames = TRUE)

#PLCO
covariates <- colnames(AllData[,110:length(AllData)])
time <- 'Overall_Survival_calculated'
event <- 'is_dead'
pth = 'C:/_research_projects/Adipocyte model project/Adipocyte analysis/PLCO/DeepLabV3plus/DeepLabV3plus fat tissue/Cox analysis'
dir.create(pth, recursive = TRUE)

res <- multiple_univariate_cox(AllData, time, event, covariates, paste0(pth, '/DLV-fat_tissue_Cox plots FATONLY'))
write.xlsx(res, file = paste0(pth,'/DLV-fat_CaFtSt_FATONLY_univariate_cox.xlsx'), rowNames = TRUE)

