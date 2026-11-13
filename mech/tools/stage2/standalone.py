"""Make a self-contained copy of the simulator (CG mesh data inlined) for sending to a phone / another PC.
Output: results/local/kingfisher_wing_fold_standalone.html (contains CC BY-SA mesh data -> not in git)."""
import pathlib
ROOT = pathlib.Path(__file__).resolve().parents[2]
html = (ROOT / 'kingfisher_wing_fold.html').read_text(encoding='utf-8')
js = (ROOT / 'ref' / 'derived' / 'ck_mesh.js').read_text(encoding='utf-8')
tag = '<script src="ref/derived/ck_mesh.js"></script>'
assert tag in html
note = ('<!-- CG body mesh: "Common Kingfisher" by Alejandro Guzmán (Sketchfab), CC BY-SA 4.0; decimated and transformed. '
        'This standalone file is therefore CC BY-SA 4.0. -->\n')
out = ROOT / 'results' / 'local' / 'kingfisher_wing_fold_standalone.html'
out.write_text(html.replace(tag, note + '<script>\n' + js + '\n</script>'), encoding='utf-8')
print(out, out.stat().st_size // 1024, 'KB')
