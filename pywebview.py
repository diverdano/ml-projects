from string import ascii_letters
from random import choice, randint
import webview
import dominate
from dominate.tags import *
from bottle import route, run, template

bootstrap = 'https://maxcdn.bootstrapcdn.com/bootstrap/3.3.6/'
bootswatch = 'https://maxcdn.bootstrapcdn.com/bootswatch/3.3.6/'
doc = dominate.document()

with doc.head:
    link(rel='stylesheet', href=bootstrap + 'css/bootstrap.min.css')
    link(rel='stylesheet', href=bootswatch + 'paper/bootstrap.min.css')
    script(src='https://code.jquery.com/jquery-2.1.1.min.js')
    [script(src=bootstrap + 'js/' + x) for x in ['bootstrap.min.js', 'bootstrap-dropdown.js']]
with doc:
    with div(cls='container'):
        with h1():
            span(cls='glyphicon glyphicon-map-marker')
            span('My Heading')
    with div(cls='row'):
        with div(cls='col-sm-6'):
            p('{{body}}')
    with div(cls='col-sm-6'):
        p('Evaluate an expression:')
        input_(id='expression', type='text')
        button('Evaluate', cls='btn btn-primary')
        div(style='margin-top: 10px;')
    with div(cls='dropdown'):
        with button(cls="btn btn-default dropdown-toggle", type="button", data_toggle='dropdown'):
            span('Dropdown')
            span(cls='caret')
            items = ('Action', 'Another action', 'Yet another action')
            ul((li(a(x, href='#')) for x in items), cls='dropdown-menu')

    with div(cls='row'):
        h3('Progress:')
        with div(cls='progress'):
            with div(cls='progress-bar', role='progressbar', style='width: 60%;'):
                span('60%')

    with div(cls='row'):
        for vid in ['4vuW6tQ0218', 'wZZ7oFKsKzY', 'NfnMJMkhDoQ']:
            with div(cls='col-sm-4'):
                with div(cls='embed-responsive embed-responsive-16by9'):
                    iframe(
                        cls='embed-responsive-item',
                        src='https://www.youtube.com/embed/' + vid,
                        frameborder='0')

@route('/')
def root():
    word = lambda: ''.join(
    choice(ascii_letters) for i in range(randint(2, 10)))
    nih_lorem = ' '.join(word() for i in range(50))
    return template(str(doc), body=nih_lorem)

if __name__ == '__main__':
    import threading
    thread = threading.Thread(
        target=run,
        kwargs=dict(host='localhost', port=8080),
        daemon=True)
    thread.start()
    webview.create_window(
        "Not a browser, honest!",
        "http://localhost:8080",
        width=800,
        height=600,
        resizable=False)
