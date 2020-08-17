# -*- coding: utf-8 -*-
import pathlib
import shutil

import keras_autodoc

PAGES = {    
    'documentation/comparator.md': [
        'teller.Comparator',
        'teller.Comparator.summary',
    ],

    'documentation/explainer.md': [
        'teller.Explainer',
        'teller.Explainer.fit',
        'teller.Explainer.summary',
    ]
}

teller_dir = pathlib.Path(__file__).resolve().parents[1]


def generate(dest_dir):
    template_dir = teller_dir / 'docs' / 'templates'

    doc_generator = keras_autodoc.DocumentationGenerator(
        PAGES,
        'https://github.com/Techtonique/teller',
        template_dir,
        #teller_dir / 'examples'
    )
    doc_generator.generate(dest_dir)

    readme = (teller_dir / 'README.md').read_text()
    index = (template_dir / 'index.md').read_text()
    index = index.replace('{{autogenerated}}', readme[readme.find('##'):])
    (dest_dir / 'index.md').write_text(index, encoding='utf-8')
    shutil.copyfile(teller_dir / 'CONTRIBUTING.md',
                    dest_dir / 'contributing.md')
    #shutil.copyfile(teller_dir / 'docs' / 'extra.css',
    #                dest_dir / 'extra.css')


if __name__ == '__main__':
    generate(teller_dir / 'docs' / 'sources')