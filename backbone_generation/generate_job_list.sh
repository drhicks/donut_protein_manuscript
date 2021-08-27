#run this after generating all of the directories with needed files
#this will create a file with each job to run
for i in $(find run*/ -maxdepth 1 -mindepth 1 -type d -name "X*") ; do echo "cd $i ; bash cmd" >> jobs.file ; done
