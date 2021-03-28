package godl

import (
	"fmt"
	"log"
	"os"

	"github.com/fatih/color"
)

func info(tmpl string, args ...interface{}) {
	msg := fmt.Sprintf(tmpl, args...)

	log.Printf("=> %s", color.GreenString(msg))
}

func warn(tmpl string, args ...interface{}) {
	msg := fmt.Sprintf(tmpl, args...)

	log.Printf("=> %s", color.YellowString(msg))
}

func failure(tmpl string, args ...interface{}) {
	msg := fmt.Sprintf(tmpl, args...)

	log.Printf("=> %s", color.RedString(msg))
}

func fatal(tmpl string, args ...interface{}) {
	msg := fmt.Sprintf(tmpl, args...)

	log.Printf("=> %s", color.RedString(msg))
	os.Exit(1)
}
