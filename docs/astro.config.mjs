// @ts-check
// The shared max-models docs setup (@max-models/starlight-theme): served at
// https://max-models.github.io/plasmacoll/, with math, the API reference and a link check.
import { defineDocs } from '@max-models/starlight-theme';

export default defineDocs({
	repo: 'plasmacoll',
	title: 'plasmacoll',
	description: 'Null-collision Monte Carlo collisions of charged particles with neutral gases.',
	pythonPackage: 'plasmacoll',
	// the method cards: a Numerics section with the overview (numerics/index.mdx) and one page per
	// module, in the order of the model, and a Numerics section on each method's API page
	methodcard: {
		pages: true,
		index: false,
		labels: {
			'plasmacoll.mcc': 'Null-collision method',
			'plasmacoll.process': 'Collision kinematics',
			'plasmacoll.background': 'Neutral backgrounds',
			'plasmacoll.cross_sections': 'Cross sections',
			'plasmacoll.rates': 'Rate coefficients',
			'plasmacoll.boltzmann': 'Two-term Boltzmann solver',
			'plasmacoll.charge_states': 'Charge states',
			'plasmacoll.beam_gas': 'Beam in gas',
			'plasmacoll.charge_exchange': 'Proton-hydrogen charge exchange',
			'plasmacoll.janev': 'Hydrogen molecule and H⁻',
			'plasmacoll.atomic_hydrogen': 'Hydrogen atoms',
			'plasmacoll.hydrogen_elastic': 'Elastic e + H, H₂',
			'plasmacoll.secondaries': 'Secondary electrons',
			'plasmacoll.argon': 'Argon',
			'plasmacoll.metals': 'Metals',
			'plasmacoll.caesium': 'Caesium',
			'plasmacoll.vibrational': 'Vibrational populations',
			'plasmacoll.population': 'Merging and splitting',
			'plasmacoll.reactor': 'Zero-dimensional reactor',
			'plasmacoll.markers': 'Markers',
		},
	},
	nav: [
		{ label: 'Getting started', link: '/getting-started/installation/' },
		{ label: 'Examples', link: '/examples/' },
		{ label: 'Numerics', link: '/numerics/' },
		{ label: 'Development', link: '/development/docs/' },
		{ label: 'API', link: '/api/plasmacoll/' },
	],
	sidebar: [
		{
			label: 'Getting started',
			items: [
				{ label: 'Installation', slug: 'getting-started/installation' },
				{ label: 'Quickstart', slug: 'getting-started/quickstart' },
			],
		},
		{
			label: 'Guides',
			items: [{ autogenerate: { directory: 'guides' } }],
		},
		{
			label: 'Examples',
			items: [{ autogenerate: { directory: 'examples' } }],
		},
		{
			label: 'Development',
			items: [
				{ label: 'Building the docs', slug: 'development/docs' },
				{ label: 'Publishing', slug: 'development/publishing' },
			],
		},
	],
});
