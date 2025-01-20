package main

// Copyright (c) 2025 Colin McRae

import (
	"encoding/json"
	"fmt"
	"github.com/predrag3141/IPSLQ/bignumber"
	"github.com/predrag3141/IPSLQ/knownanswertest"
	"github.com/predrag3141/IPSLQ/pslqops"
	"os"
	"time"
)

const (
	minDimension                    = 10
	dimensionIncr                   = 10
	maxDimension                    = 40 // 100
	numTests                        = 10
	relationElementRange            = 5
	randomRelationProbabilityThresh = 0.001
	maxIterations                   = 20000
	bigNumberPrecision              = 1500
)

func main() {
	if len(os.Args) != 2 {
		fmt.Println("Usage: go run main.go base_directory")
		return
	}
	err := bignumber.Init(bigNumberPrecision)
	if err != nil {
		fmt.Printf("Could not initialize bignumber: %q", err.Error())
	}
	for testNbr := 0; testNbr < numTests; testNbr++ {
		for dim := minDimension; dim <= maxDimension; dim += dimensionIncr {
			err = oneTest(testNbr, dim, os.Args[1], "main")
			if err != nil {
				fmt.Printf("%q", err.Error())
				return
			}
		}
	}
}

func oneTest(testNbr, dim int, baseDirectory, caller string) error {
	// Initializations
	caller = fmt.Sprintf("%s-oneTest", caller)

	// Open the results file
	fileName := fmt.Sprintf(
		"%s/%s/test_%d-dim_%d",
		baseDirectory, time.Now().Format("2006_01_02"), testNbr, dim,
	)
	file, err := os.OpenFile(fileName, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	defer file.Close()

	// Create the PSLQ context
	var pc *knownanswertest.PSLQContext
	pc = knownanswertest.NewPSLQContext(
		dim, relationElementRange, randomRelationProbabilityThresh,
	)

	// Create the PSLQ state
	var state *pslqops.State
	state, err = pslqops.NewState(pc.InputAsDecimalString)

	// Run PSLQ
	for numIterations := 0; numIterations < maxIterations; numIterations++ {
		var terminated bool
		terminated, err = state.OneIteration(pslqops.NextIntOp)
		if err != nil {
			fmt.Printf("Could not perform iteration %d: %q", numIterations, err.Error())
			return fmt.Errorf("err")
		}
		err = pc.UpdateSolutions(
			state, numIterations, terminated || (numIterations == maxIterations-1),
		)
		if err != nil {
			return err
		}
		if terminated {
			break
		}
	}

	// Write results
	var resultsAsByteArray []byte
	resultsAsByteArray, err = json.Marshal(pc)
	err = writeResults(file, string(resultsAsByteArray), caller)
	if err != nil {
		return fmt.Errorf(
			"%s: could not write results to %s: %q",
			caller, fileName, err.Error(),
		)
	}
	return nil
}

func writeResults(file *os.File, results, caller string) error {
	caller = fmt.Sprintf("%s-writeResults", caller)

	// Write the string to the file
	_, err := file.WriteString(results)
	if err != nil {
		return fmt.Errorf("%s: Error writing to file: %q", caller, err)
	}
	return nil
}
