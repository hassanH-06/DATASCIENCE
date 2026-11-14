args <- commandArgs(trailingOnly=FALSE)
script <- sub('^--file=', '', args[grepl('^--file=',args)])
if(length(script)) setwd(dirname(dirname(normalizePath(script))))
if(!requireNamespace('rmarkdown',quietly=TRUE)) stop('Install rmarkdown: install.packages("rmarkdown")')
if(!rmarkdown::pandoc_available()) {
 p<-file.path(Sys.getenv('LOCALAPPDATA'),'Programs/RStudio/resources/app/bin/quarto/bin/tools')
 if(file.exists(file.path(p,'pandoc.exe')))Sys.setenv(RSTUDIO_PANDOC=p)
}
rmarkdown::render('Milestone_1.Rmd',output_file='Milestone_1.html',envir=new.env())
