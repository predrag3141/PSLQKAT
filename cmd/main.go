package main

// Copyright (c) 2025 Colin McRae

import (
	"fmt"
	"github.com/predrag3141/IPSLQ/bignumber"
	"github.com/predrag3141/IPSLQ/knownanswertest"
	"github.com/predrag3141/IPSLQ/pslqops"
	"os"
	"strconv"
)

const (
	relationElementRange            = 5
	randomRelationProbabilityThresh = 0.001
	maxIterations                   = 50000
	bigNumberPrecision              = 1500
	minDimensionName                = "min_dim"
	dimensionIncrName               = "dim_incr"
	maxDimensionName                = "max_dim"
	numTestsName                    = "num_tests"
)

func main() {
	// Check argument count
	if len(os.Args) != 6 {
		fmt.Printf(
			"Usage: go run main.go base_directory %s %s %s %s\n",
			minDimensionName, dimensionIncrName, maxDimensionName, numTestsName,
		)
		return
	}

	// Parse arguments
	var err error
	var minDimensionAsInt, dimensionIncrAsInt, maxDimensionAsInt, numTestsAsInt int
	baseDirectory := os.Args[1]
	minDimensionAsStr := os.Args[2]
	dimensionIncrAsStr := os.Args[3]
	maxDimensionAsStr := os.Args[4]
	numTestsAsStr := os.Args[5]
	minDimensionAsInt, err = strconv.Atoi(minDimensionAsStr)
	if err != nil {
		fmt.Printf("Could not convert %s = %s to integer", minDimensionName, minDimensionAsStr)
		return
	}
	dimensionIncrAsInt, err = strconv.Atoi(dimensionIncrAsStr)
	if err != nil {
		fmt.Printf("Could not convert %s = %s to integer", dimensionIncrName, dimensionIncrAsStr)
		return
	}
	maxDimensionAsInt, err = strconv.Atoi(maxDimensionAsStr)
	if err != nil {
		fmt.Printf("Could not convert %s = %s to integer", maxDimensionName, maxDimensionAsStr)
		return
	}
	numTestsAsInt, err = strconv.Atoi(numTestsAsStr)
	if err != nil {
		fmt.Printf("Could not convert %s = %s to integer", numTestsName, numTestsAsStr)
		return
	}

	// Initialize big number precision
	err = bignumber.Init(bigNumberPrecision)
	if err != nil {
		fmt.Printf("Could not initialize bignumber: %q", err.Error())
		return
	}

	// Run tests
	for testNbr := 0; testNbr < numTestsAsInt; testNbr++ {
		for dim := minDimensionAsInt; dim <= maxDimensionAsInt; dim += dimensionIncrAsInt {
			err = oneTest(dim, baseDirectory, "main")
			if err != nil {
				fmt.Printf("%q", err.Error())
				return
			}
		}
	}
}

func oneTest(dim int, baseDirectory, caller string) error {
	// Initializations
	caller = fmt.Sprintf("%s-oneTest", caller)
	katLog, err := knownanswertest.NewKATLog(baseDirectory, dim, 100, 25)

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
		err = pc.Update(state, terminated || (numIterations == maxIterations-1))
		if err != nil {
			return fmt.Errorf("%s: could not update the PSLQ context: %q", caller, err.Error())
		}
		err = katLog.ReportProgress(pc)
		if err != nil {
			return fmt.Errorf("%s: could not report progress: %q", caller, err.Error())
		}
		if terminated {
			break
		}
	}

	// Write results
	err = katLog.ReportResults(pc)
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
