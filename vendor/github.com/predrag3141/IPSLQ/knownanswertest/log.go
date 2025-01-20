package knownanswertest

// Copyright (c) 2025 Colin McRae

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"time"
)

const (
	progressDirName                 = "progress"
	resultsDirName                  = "results"
	timeSinceStartHeader            = "time since start"
	iterationsBeforeInvertingHeader = "iterations before inverting"
	iterationsAfterInvertingHeader  = "iterations after inverting"
	totalIterationsHeader           = "total iterations"
	foundRelationHeader             = "found relation"
)

// KATLog holds the state of a known answer test logger
type KATLog struct {
	progressFile                   *os.File
	resultFile                     *os.File
	progressFilePath               string
	resultFilePath                 string
	reportingPeriodBeforeInverting int
	reportingPeriodAfterInverting  int // separate value after inverting when PSLQ runs more slowly
	startTime                      time.Time
}

// NewKATLog creates an instance of the KAT logger
func NewKATLog(
	baseDir string, dimension,
	reportingPeriodBeforeInverting, reportingPeriodAfterInverting int,
) (*KATLog, error) {
	// Scalar initializations
	retVal := &KATLog{}
	var err error
	retVal.startTime = time.Now()
	timeStamp := retVal.startTime.Format("2006_01_02T15_04_05")
	retVal.reportingPeriodBeforeInverting = reportingPeriodBeforeInverting
	retVal.reportingPeriodAfterInverting = reportingPeriodAfterInverting

	// Progress directory
	for _, dirName := range []string{progressDirName, resultsDirName} {
		err = createDirectory(filepath.Join(baseDir, dirName), "NewKATLog")
		if err != nil {
			return nil, err
		}
	}

	// Progress file
	retVal.progressFilePath = filepath.Join(
		baseDir, progressDirName, fmt.Sprintf("dim_%d-%s", dimension, timeStamp),
	)
	retVal.progressFile, err = os.OpenFile(
		retVal.progressFilePath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644,
	)
	if err != nil {
		return nil, fmt.Errorf(
			"NewKATLog: could not open %s: %q", retVal.progressFilePath, err.Error())
	}

	// Results file
	retVal.resultFilePath = filepath.Join(
		baseDir, resultsDirName, fmt.Sprintf("dim_%d-%s", dimension, timeStamp),
	)
	retVal.resultFile, err = os.OpenFile(
		retVal.resultFilePath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644,
	)
	if err != nil {
		return nil, fmt.Errorf(
			"NewKATLog: could not open %s: %q", retVal.resultFilePath, err.Error(),
		)
	}
	return retVal, nil
}

func (kl *KATLog) ReportProgress(pc *PSLQContext) error {
	// Write the header
	if pc.TotalIterations == 0 {
		_, err := kl.progressFile.WriteString(fmt.Sprintf(
			"%s,%s,%s,%s,%s\n",
			timeSinceStartHeader, iterationsBeforeInvertingHeader, iterationsAfterInvertingHeader,
			totalIterationsHeader, foundRelationHeader,
		))
		if err != nil {
			return fmt.Errorf(
				"ReportProgress: could not write header to %s: %q",
				kl.progressFilePath, err.Error(),
			)
		}
		return nil // Skip the report at iteration 0, other than writing the header
	}

	// Determine whether to report progress
	reportProgress := false
	if pc.IterationsAfterInverting == 0 {
		if (pc.TotalIterations % kl.reportingPeriodBeforeInverting) == 0 {
			reportProgress = true
		}
	} else if (pc.TotalIterations % kl.reportingPeriodAfterInverting) == 0 {
		reportProgress = true
	}
	if !reportProgress {
		return nil
	}

	// Report progress
	_, err := kl.progressFile.WriteString(fmt.Sprintf("%v,%d,%d,%d,%v\n",
		time.Since(kl.startTime), pc.IterationsBeforeInverting, pc.IterationsAfterInverting,
		pc.TotalIterations, pc.FoundRelation,
	))
	if err != nil {
		return fmt.Errorf(
			"ReportProgress: could not write progress to %s: %q",
			kl.progressFilePath, err.Error(),
		)
	}
	return nil
}

func (kl *KATLog) ReportResults(pc *PSLQContext) error {
	resultsAsByteArray, err := json.Marshal(pc)
	if err != nil {
		return fmt.Errorf(
			"ReportResults: could not write results to %s: %q",
			kl.resultFilePath, err.Error(),
		)
	}
	_, err = kl.resultFile.WriteString(string(resultsAsByteArray))
	if err != nil {
		return fmt.Errorf(
			"ReportResults: could not write results to %s: %q",
			kl.resultFilePath, err.Error(),
		)
	}
	return nil
}

func createDirectory(directoryPath, caller string) error {
	caller = fmt.Sprintf("%s-createDirectory", caller)
	_, err := os.Stat(directoryPath)
	if os.IsNotExist(err) {
		// Directory does not exist, create it
		err = os.Mkdir(directoryPath, 0755)
		if err != nil {
			return fmt.Errorf(
				"%s: could not create directory %s: %q", caller, directoryPath, err,
			)
		}
	}
	if err != nil {
		return fmt.Errorf(
			"%s: could not stat directory %s before making it: %q", caller, directoryPath, err,
		)
	}
	_, err = os.Stat(directoryPath)
	if err != nil {
		return fmt.Errorf(
			"%s: could not stat directory %s after making it: %q",
			caller, directoryPath, err.Error(),
		)
	}
	return nil
}
