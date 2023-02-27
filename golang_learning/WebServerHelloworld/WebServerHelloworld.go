package WebServerHelloworld

import (
	"fmt"
	"html/template"
	"log"
	"net/http"
	"time"

	"example.com/golang_learning/MysqlTest"
	"github.com/gorilla/mux"
	"github.com/gorilla/sessions"
	"github.com/gorilla/websocket"
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

	/*
	   A basic HTTP server has a few key jobs to take care of.

	   Process dynamic requests: Process incoming requests from users who browse the website, log into their accounts or post images.

	   Serve static assets: Serve JavaScript, CSS and images to browsers to create a dynamic experience for the user.

	   Accept connections: The HTTP Server must listen on a specific port to be able to accept connections from the internet.
	*/

	// Process dynamic requests:
	/*
		For the dynamic aspect, the http.Request contains all information about the request and it’s parameters.
		You can read GET parameters with r.URL.Query().Get("token") or POST parameters (fields from an HTML form) with r.FormValue("email").
	*/
	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintf(w, "Welcome to my Website! The URL.Query().Get(\"token\") is:%v", r.URL.Query().Get("token"))
	})

	// Serve static assets:
	/*
		To serve static assets like JavaScript, CSS and images, we use the inbuilt http.FileServer and point it to a url path.
		For the file server to work properly it needs to know, where to serve files from.
	*/
	fs := http.FileServer(http.Dir("static/"))
	/*
		Once our file server is in place, we just need to point a url path at it, just like we did with the dynamic requests.
		One thing to note: In order to serve files correctly, we need to strip away a part of the url path.
		Usually this is the name of the directory our files live in.
	*/
	http.Handle("/static/", http.StripPrefix("/static/", fs))

	// Accept connections:
	http.ListenAndServe("127.0.0.1:8080", nil)
}

// Routing (Using gorilla/mux)
func RouteTest() {
	// Creating a new Router
	/*
		First create a new request router.
		The router is the main router for your web application and will later be passed as parameter to the server.
		It will receive all HTTP connections and pass it on to the request handlers you will register on it.
	*/
	r := mux.NewRouter()

	// Registering a Request Handler && URL Parameters
	/*
		The biggest strength of the gorilla/mux Router is the ability to extract segments from the request URL. As an example, this is a URL in your application:

		/books/go-programming-blueprint/page/10
		This URL has two dynamic segments:
		Book title slug (go-programming-blueprint)
		Page (10)
		To have a request handler match the URL mentioned above you replace the dynamic segments of with placeholders in your URL pattern like so:
	*/
	r.HandleFunc("/books/{title}/page/{page}", func(w http.ResponseWriter, r *http.Request) {
		vars := mux.Vars(r)
		title := vars["title"]
		page := vars["page"]

		fmt.Fprintf(w, "You've requested the book: %s on page %s\n", title, page)
	})

	// Setting the HTTP server's router
	/*
		Ever wondered what the nil in http.ListenAndServe(":80", nil) ment?
		It is the parameter for the main router of the HTTP server.
		By default it’s nil, which means to use the default router of the net/http package.
		To make use of your own router, replace the nil with the variable of your router r.
	*/
	http.ListenAndServe("127.0.0.1:8080", r)
}

func First_Template() {
	type Todo struct {
		Title string
		Done  bool
	}
	type TodoPageData struct {
		PageTitle string
		Todos     []Todo
	}

	tmpl := template.Must(template.ParseFiles("static/layout.html"))
	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		data := TodoPageData{
			PageTitle: `This example shows a TODO list, written as an unordered list (ul) in HTML. 
			When rendering templates, the data passed in can be any kind of Go's data structures.
			To access the data in a template the top most variable is access by {{.}}. 
			The dot inside the curly braces is called the pipeline and the root element of the data. 
			`,
			Todos: []Todo{
				{Title: "Task 1", Done: false},
				{Title: "Task 2", Done: false},
				{Title: "Task 3", Done: true},
			},
		}
		tmpl.Execute(w, data)
	})

	http.ListenAndServe("127.0.0.1:8080", nil)
}

func Assets_and_Files() {
	fs := http.FileServer(http.Dir("assets/"))
	http.Handle("/static/", http.StripPrefix("/static/", fs))
	http.ListenAndServe("127.0.0.1:8080", nil)
}

func Forms_exp() {
	type ContactDetails struct {
		Email   string
		Subject string
		Message string
	}

	tmpl := template.Must(template.ParseFiles("static/forms.html"))

	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			tmpl.Execute(w, nil)
			return
		}

		detail := ContactDetails{
			Email:   r.FormValue("email"),
			Subject: r.FormValue("subject"),
			Message: r.FormValue("message"),
		}

		// do something with details
		var get_data = detail
		err := MysqlTest.DataFromWeb(get_data.Email, get_data.Subject, get_data.Message)
		if err != nil {
			log.Print(err)
			tmpl.Execute(w, struct{ Success bool }{false})
		} else {
			tmpl.Execute(w, struct{ Success bool }{true})
		}

	})

	http.ListenAndServe("127.0.0.1:8080", nil)
}

// Middleware (Basic)
// This example will show how to create basic logging middleware in Go.
// A middleware simply takes a http.HandlerFunc as one of its parameters, wraps it and returns a new http.HandlerFunc for the server to call.
func logging(f http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		log.Println(r.URL.Path)
		f(w, r)
	}
}
func foo(w http.ResponseWriter, r *http.Request) {
	fmt.Fprintln(w, "foo")
}
func bar(w http.ResponseWriter, r *http.Request) {
	fmt.Fprintln(w, "bar")
}
func Basic_Middleware() {
	const msg = `
	Middleware (Basic)
	This example will show how to create basic logging middleware in Go.
	A middleware simply takes a http.HandlerFunc as one of its parameters, wraps it and returns a new http.HandlerFunc for the server to call.
	try to query '/foo' or '/bar'
	`
	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintln(w, msg)
	})
	http.HandleFunc("/foo", logging(foo))
	http.HandleFunc("/bar", logging(bar))

	http.ListenAndServe("127.0.0.1:8080", nil)
}

// Middleware (Advanced)
/*
This example will show how to create a more advanced version of middleware in Go.

Here we define a new type Middleware which makes it eventually easier to chain multiple middlewares together.

This idea is inspired by Mat Ryers’ talk about Building APIs. You can find a more detailed explaination including the talk here.
https://medium.com/@matryer/writing-middleware-in-golang-and-how-go-makes-it-so-much-fun-4375c1246e81

This snippet explains in detail how a new middleware is created. In the full example below, we reduce this version by some boilerplate code.
*/
type middleware func(http.HandlerFunc) http.HandlerFunc

/*
template:
func createNewMiddleware() Middleware {

    // Create a new Middleware
    middleware := func(next http.HandlerFunc) http.HandlerFunc {

        // Define the http.HandlerFunc which is called by the server eventually
        handler := func(w http.ResponseWriter, r *http.Request) {

            // ... do middleware things

            // Call the next middleware/handler in chain
            next(w, r)
        }

        // Return newly created handler
        return handler
    }

    // Return newly created middleware
    return middleware
}
*/

// logging logs all requests with its path and the time it took to process
func loggingV2() middleware {

	// create a new middleware
	return func(hf http.HandlerFunc) http.HandlerFunc {
		// define the http.HandlerFunc
		return func(w http.ResponseWriter, r *http.Request) {
			// do middleware things
			start := time.Now()
			defer func() {
				log.Println(r.URL.Path, time.Since(start))
			}()
			// call the next middleware/handler in chain
			hf(w, r)
		}
	}
}

// method ensures that url can only be requested with a specific method, else return a 400 bad request
func method(m string) middleware {
	// create a middleware
	return func(f http.HandlerFunc) http.HandlerFunc {
		fmt.Printf("type:%T,value:%v\n", f, f)
		// define the http.HandlerFunc
		return func(w http.ResponseWriter, r *http.Request) {
			// do middleware things
			if r.Method != m {
				http.Error(w, http.StatusText(http.StatusBadRequest), http.StatusBadRequest)
				return
			}
			// call the next middleware/handler in chain
			f(w, r)
		}

	}
}

// chain applies middlewares to a http.HandlerFunc
func chain(f http.HandlerFunc, middlewares ...middleware) http.HandlerFunc {
	fmt.Printf("hello: type:%T,value:%v", f, f)
	for _, m := range middlewares {
		/*
			Our 'chain' function will simply iterate over all middlewares,
			calling them one by one (in reverse order) in a chained manner, returning the result of the first middleware.
		*/
		// middlewares[0]=method
		// middlewares[1]=loggingv2
		fmt.Printf("type:%T, value:%v\n", m, m)
		f = m(f)
		/* 为什么可以在method中得到 f 的值？
		传入 []middleware 中的确实是 method("GET"), 可以传入的原因是该函数返回了 func(http.HandlerFunc) http.HandlerFunc,
		也就是说 []middleware 存的是 method 函数的 func(http.HandlerFunc) http.HandlerFunc。
		当执行 m(f) 时, 执行的是 method 函数的 func(http.HandlerFunc) http.HandlerFunc, 也就解释了为什么可以得到 f, 即 hello()。
		*/
	}
	return f // 此时 f=loggingv2(method(hello))
}
func hello(w http.ResponseWriter, r *http.Request) {
	fmt.Fprintln(w, "hello, world!")
}
func Advanced_Middleware() {
	ms := []middleware{}
	ms = append(ms, method("?"))
	a := ms[0](hello)
	fmt.Printf("type:%T, value:%v", a, a)
	http.HandleFunc("/", chain(hello, method("GET"), loggingV2()))
	http.ListenAndServe("127.0.0.1:8080", nil)
}

// Session
/*
In this example we will only allow authenticated users to view our secret message on the /secret page.
To get access to it, the will first have to visit /login to get a valid session cookie, which logs him in.
Additionally he can visit /logout to revoke his access to our secret message.
*/
var (
	key   = []byte("super-secret-key")
	store = sessions.NewCookieStore(key)
)

func secret(w http.ResponseWriter, r *http.Request) {
	session, _ := store.Get(r, "cookie-name")

	// check if user is authenticated
	if auth, ok := session.Values["authenticated"].(bool); !ok || !auth {
		http.Error(w, "Forbidden", http.StatusForbidden)
		return
	}
	// print secret message
	fmt.Fprintln(w, "The cake is a lie!")
}
func login(w http.ResponseWriter, r *http.Request) {
	session, _ := store.Get(r, "cookie-name")
	// Authentication goes here
	// ....

	// set user as authenticated
	session.Values["authenticated"] = true
	session.Save(r, w)
}
func logout(w http.ResponseWriter, r *http.Request) {
	session, _ := store.Get(r, "cookie-name")
	// revoke users authentication
	session.Values["authenticated"] = false
	session.Save(r, w)
}
func Session_test() {
	http.HandleFunc("/secret", secret)
	http.HandleFunc("/login", login)
	http.HandleFunc("/logout", logout)
	http.ListenAndServe("127.0.0.1:8080", nil)
}

// Websockets
func WebSockets_test() {
	var upgrader = websocket.Upgrader{
		ReadBufferSize:  1024,
		WriteBufferSize: 1024,
	}

	http.HandleFunc("/echo", func(w http.ResponseWriter, r *http.Request) {
		conn, _ := upgrader.Upgrade(w, r, nil) // error ignored for sake of simplicity
		for {
			// read message from browser
			msgType, msg, err := conn.ReadMessage()
			if err != nil {
				return
			}

			//print the message to the console
			fmt.Printf("%s sent: %s\n", conn.RemoteAddr(), string(msg))

			//write message back to browser
			if err = conn.WriteMessage(msgType, msg); err != nil {
				return
			}
		}
	})

	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		http.ServeFile(w, r, "static/websockets.html")
	})

	http.ListenAndServe("127.0.0.1:8080", nil)
}
