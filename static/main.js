window.onload = async function () {
    var cell_ind = 0;
    doc = [];
    for (let i = 1; i <= 9; i++) {
        doc.push(document.getElementById(i));
    }
    for (let i = 0; i < doc.length; i++) {
        $("#" + (i + 1)).unbind("click");
        doc[i].onclick = async function () {
            cell_ind = $(this).attr('id')
            con = await eel.get_move_num()()
            anime({
                targets: this,
                scaleX: [
                    {value: 0.6, duration: 200, easing: 'linear'},
                    {value: 0.1, duration: 250,easing: 'linear'},
                    {value: 0.6, duration: 250, easing: 'linear'},
                    {value: 1, duration: 300,easing: 'linear'}]

            }).finished.then(function (){
                          if (con % 2 == 0) {
                $('#'+(i+1)).css({
                    'background': 'url(nolik.png) no-repeat',
                    'background-size': '100% 100%',
                    'object-fit': 'fill;'
                });
            } else {
                $('#'+(i+1)).css({
                    'background': 'url(crestik.png) no-repeat',
                    'background-size': '100% 100%',
                    'object-fit': 'fill;'
                });
            }
            })


        }
    }
    eel.expose(f);

    function f() {
        return cell_ind;
    }
}