package tabnet

import (
	"log"
	"os"
)

var (
	testLogger *log.Logger
)

func init() {
	f, err := os.Create("test.log")
	if err != nil {
		panic(err)
	}

	testLogger = log.New(f, "[G]", log.LstdFlags)
}
