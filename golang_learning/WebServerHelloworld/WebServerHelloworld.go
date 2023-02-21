package WebServerHelloworld

import (
	"fmt"
	"net/http"
)

func WebServerHello() {
	// registering a request handler
	// when someone browses your websites(http://example.com/),he or she will be greeted as follow message.
	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintf(w, "Hello,you've requested:%s\n", r.URL.Path)
	})

	// listen for http connections
	// The request handler alone can not accept any HTTP connections from the outside.
	// An HTTP server has to listen on a port to pass connections on to the request handler.
	http.ListenAndServe("127.0.0.1:8080", nil)
}

func HttpServer() {

}
