package deepzen

import (
	"fmt"

	"github.com/fatih/color"
)

// HandleErr panics if the given err is not nil
func HandleErr(err error, where string, args ...interface{}) {
	if err == nil {
		return
	}

	message := fmt.Sprintf(where, args...)

	panic(fmt.Sprintf("%s: %v", color.RedString(message), err))
}

func ErrorF(lt LayerType, template string, args ...interface{}) error {
	args = append([]interface{}{lt}, args...)
	return fmt.Errorf("[%s] "+template, args...)
}
